"""aeon.desktop.runtime — AeonDesktopRuntime.

Authentic Aeon token-generation runtime for the 7M P2 research preview.
Every generated token executes through aeon.hybrid.HybridModel via the
same authoritative path used by the P2 training run — no substitution,
no fallback impersonation, no cloud call.

Design constraints (per §3, §11-§16, §19, §21):

  * K = 16 fixed; recursion fp32; substrate autonomous; ACIS OFF.
  * One shared broadcast per boundary.
  * Session state is EPHEMERAL and per-session (Policy B — deterministic
    history replay). Every generation reconstructs Recursion + Substrate
    state through the authentic forward path from the session's token
    history. No hidden-state persistence is claimed.
  * Cancellation is checked between token steps. The Stop button never
    kills the runtime.
  * Local-only; no network transport; no arbitrary deserialization.
  * All release identities verified against the release manifest before
    load; refuses fail-closed on any mismatch.
  * Streams RuntimeEvents via a caller-supplied handler.

This runtime is in-process. The Tkinter chat UI (aeon.desktop.chat_ui)
runs the runtime on a background thread and consumes events from the
event queue.
"""
from __future__ import annotations

import hashlib
import json
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from .protocol import (
    BOUNDS, ErrorCode, EventKind, GenerationOptions, RuntimeEvent,
    RuntimeState, SettingsInvalid, is_valid_transition, make_event,
    validate_settings,
)


# Limits (per §15)
MAX_PROMPT_BYTES = 16 * 1024
MAX_PROMPT_CHARS = 8 * 1024
MAX_PROMPT_TOKENS = 2048
MAX_SESSION_TOKENS = 4096
MAX_CONCURRENT_SESSIONS = 16
MAX_CONCURRENT_GENERATION_REQUESTS = 1
MAX_QUEUED_EVENTS = 4096


class RuntimeError_(RuntimeError):
    def __init__(self, code: ErrorCode, detail: str = ""):
        super().__init__(f"{code.value}: {detail}" if detail else code.value)
        self.code = code
        self.detail = detail


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
@dataclass
class SessionState:
    session_id: str
    token_history: List[int] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_used_at: float = field(default_factory=time.time)
    closed: bool = False


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
class AeonDesktopRuntime:
    """The authoritative chat runtime.

    Usage:
        rt = AeonDesktopRuntime(event_handler=my_handler)
        rt.load_release(Path("release-assets/aeon-desktop-p2-proxy"))
        sid = rt.create_session()
        rid = rt.submit_prompt(sid, "hello", GenerationOptions())
        # events stream to my_handler
        rt.shutdown()
    """

    def __init__(self, event_handler: Optional[Callable[[RuntimeEvent], None]] = None):
        self._state = RuntimeState.NOT_STARTED
        self._state_lock = threading.RLock()
        self._event_seq = 0
        self._event_handler = event_handler
        self._model = None
        self._tokenizer = None
        self._release_root: Optional[Path] = None
        self._release_manifest: Optional[Dict] = None
        self._sessions: Dict[str, SessionState] = {}
        self._sessions_lock = threading.RLock()
        self._active_request_id: Optional[str] = None
        self._cancel_event = threading.Event()
        self._active_generation_thread: Optional[threading.Thread] = None
        self._last_gen_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # State machine
    # ------------------------------------------------------------------
    def state(self) -> RuntimeState:
        with self._state_lock:
            return self._state

    def _transition(self, nxt: RuntimeState) -> None:
        with self._state_lock:
            if not is_valid_transition(self._state, nxt):
                raise RuntimeError_(
                    ErrorCode.RUNTIME_START_FAILED,
                    f"invalid state transition {self._state.value} -> {nxt.value}")
            self._state = nxt

    def _emit(self, kind: EventKind, *, session_id: Optional[str] = None,
                request_id: Optional[str] = None, payload: Optional[Dict] = None) -> RuntimeEvent:
        self._event_seq += 1
        ev = make_event(kind, seq=self._event_seq, session_id=session_id,
                            request_id=request_id, payload=payload)
        if self._event_handler is not None:
            try:
                self._event_handler(ev)
            except Exception:
                pass  # never let a handler failure kill the runtime
        return ev

    # ------------------------------------------------------------------
    # Preflight + load
    # ------------------------------------------------------------------
    def preflight(self) -> Dict[str, Any]:
        self._transition(RuntimeState.STARTING)
        self._emit(EventKind.RUNTIME_STARTING)
        self._transition(RuntimeState.PREFLIGHT)
        self._emit(EventKind.RUNTIME_PREFLIGHT_STARTED)
        info = {
            "torch_version": torch.__version__,
            "device": "cpu",
            "cuda_available": bool(torch.cuda.is_available()),
            "num_threads": torch.get_num_threads(),
            "network_policy": "offline_only",
        }
        self._emit(EventKind.RUNTIME_PREFLIGHT_COMPLETED, payload=info)
        return info

    def load_release(self, release_root: Path) -> Dict[str, Any]:
        release_root = Path(release_root).resolve()
        self._transition(RuntimeState.VALIDATING_RELEASE)
        self._emit(EventKind.MODEL_VALIDATION_STARTED,
                     payload={"release_root": str(release_root)})

        # 1. Manifest present + parseable
        rel_path = release_root / "manifests" / "release_manifest.json"
        if not rel_path.exists():
            self._fail(ErrorCode.RELEASE_MANIFEST_MISSING, str(rel_path))
        try:
            manifest = json.loads(rel_path.read_text(encoding="utf-8"))
        except Exception as e:
            self._fail(ErrorCode.RELEASE_MANIFEST_INVALID, str(e))
        # 2. schema
        if int(manifest.get("release_schema_version", 0)) != 1:
            self._fail(ErrorCode.RELEASE_MANIFEST_INVALID,
                          f"unsupported schema version {manifest.get('release_schema_version')}")
        # 3. fixed_k
        if int(manifest.get("fixed_k", -1)) != 16:
            self._fail(ErrorCode.FIXED_K_MISMATCH,
                          f"fixed_k={manifest.get('fixed_k')} in manifest, expected 16")

        # 4. Model file present + digest matches
        model_rel = manifest["model_artifact_path"]
        model_path = (release_root / model_rel).resolve()
        if not str(model_path).startswith(str(release_root)):
            self._fail(ErrorCode.RELEASE_MANIFEST_INVALID,
                          "model path escapes bundle root")
        if not model_path.exists():
            self._fail(ErrorCode.MODEL_MISSING, str(model_path))
        want_model_sha = manifest["model_artifact_sha256"]
        got_model_sha = _sha256_file(model_path)
        if got_model_sha != want_model_sha:
            self._fail(ErrorCode.MODEL_DIGEST_MISMATCH,
                          f"got {got_model_sha}, want {want_model_sha}")

        # 5. Tokenizer file present + digest matches + vocab matches
        tok_rel = manifest["tokenizer_model_path"]
        tok_path = (release_root / tok_rel).resolve()
        if not tok_path.exists():
            self._fail(ErrorCode.TOKENIZER_MISSING, str(tok_path))
        got_tok_sha = _sha256_file(tok_path)
        if got_tok_sha != manifest["tokenizer_model_sha256"]:
            self._fail(ErrorCode.TOKENIZER_DIGEST_MISMATCH,
                          f"got {got_tok_sha}")

        # 6. Instantiate tokenizer + verify vocab + special ids
        from aeon.tokenizer import AeonTokenizer, PAD_ID, UNK_ID, BOS_ID, EOS_ID
        tok = AeonTokenizer(str(tok_path))
        if tok.vocab_size != int(manifest["tokenizer_vocab_size"]):
            self._fail(ErrorCode.TOKENIZER_VOCAB_MISMATCH,
                          f"loader vocab {tok.vocab_size} != manifest {manifest['tokenizer_vocab_size']}")
        for k, want in manifest["special_token_ids"].items():
            got = getattr(tok, k)
            if int(got) != int(want):
                self._fail(ErrorCode.TOKENIZER_VOCAB_MISMATCH,
                              f"{k}: loader {got} != manifest {want}")

        # 7. Config + architecture digest
        import yaml
        cfg_path = release_root / manifest.get(
            "config_relpath", "config/aeon_lbc1_proxy.yaml")
        # release_manifest omits config_relpath in v1; fall back to
        # architecture_manifest.
        arch_path = release_root / "manifests" / "architecture_manifest.json"
        arch_manifest = json.loads(arch_path.read_text(encoding="utf-8"))
        cfg_relpath = arch_manifest.get("config_relpath", "config/aeon_lbc1_proxy.yaml")
        cfg_path = (release_root / cfg_relpath).resolve()
        cfg = yaml.safe_load(open(cfg_path))
        cfg_digest = "sha256:" + hashlib.sha256(
            json.dumps(cfg["model"], sort_keys=True).encode("utf-8")).hexdigest()
        if cfg_digest != manifest["model_configuration_digest"]:
            self._fail(ErrorCode.MODEL_CONFIG_MISMATCH,
                          f"cfg_digest {cfg_digest} != manifest {manifest['model_configuration_digest']}")

        # 8. Build model with same config + strict-load exported weights
        self._emit(EventKind.MODEL_VALIDATION_COMPLETED)
        self._transition(RuntimeState.LOADING_MODEL)
        self._emit(EventKind.MODEL_LOADING_STARTED,
                     payload={"parameter_count": int(manifest["parameter_count"])})
        from aeon.hybrid import HybridModel
        from aeon.transformer import AeonTransformerConfig
        mcfg = cfg["model"]; tcfg = mcfg["transformer"]
        tconfig = AeonTransformerConfig(
            vocab_size=int(manifest["tokenizer_vocab_size"]),
            hidden_size=tcfg["hidden_size"],
            num_hidden_layers=tcfg["num_hidden_layers"],
            num_attention_heads=tcfg["num_attention_heads"],
            num_key_value_heads=tcfg["num_key_value_heads"],
            head_dim=tcfg["head_dim"],
            intermediate_size=tcfg["intermediate_size"],
            max_position_embeddings=tcfg["max_position_embeddings"])
        model = HybridModel(transformer_config=tconfig, h_rec=mcfg["h_rec"],
                                K=int(mcfg["K"]), margin_h=mcfg["margin_h"],
                                margin_c=mcfg["margin_c"], use_embedding_input=True,
                                dtype=torch.float32).to(dtype=torch.float32)
        n_params = sum(p.numel() for p in model.parameters())
        if n_params != int(manifest["parameter_count"]):
            self._fail(ErrorCode.MODEL_CONFIG_MISMATCH,
                          f"built {n_params} params, manifest {manifest['parameter_count']}")
        if int(getattr(model, "K", -1)) != 16:
            self._fail(ErrorCode.FIXED_K_MISMATCH, f"HybridModel.K={model.K}")
        # weights_only=True is critical safety — refuses arbitrary classes.
        try:
            state = torch.load(str(model_path), map_location="cpu", weights_only=True)
        except Exception as e:
            self._fail(ErrorCode.MODEL_SCHEMA_UNSUPPORTED,
                          f"weights_only load failed: {e}")
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            self._fail(ErrorCode.MODEL_LOAD_FAILED, f"missing keys: {missing[:3]}...")
        if unexpected:
            self._fail(ErrorCode.MODEL_LOAD_FAILED, f"unexpected keys: {unexpected[:3]}...")
        model.eval()

        self._model = model
        self._tokenizer = tok
        self._release_root = release_root
        self._release_manifest = manifest
        self._transition(RuntimeState.READY)
        self._emit(EventKind.MODEL_READY, payload={
            "release_id": manifest["release_id"],
            "release_label": manifest["release_label"],
            "tested_scale": manifest["tested_scale"],
            "parameter_count": n_params,
            "K": 16,
            "ACIS_default": manifest["ACIS_default"],
        })
        return {"release_id": manifest["release_id"],
                  "parameter_count": n_params}

    def _fail(self, code: ErrorCode, detail: str) -> None:
        with self._state_lock:
            self._state = RuntimeState.FAILED
        self._emit(EventKind.RUNTIME_FAILED,
                     payload={"code": code.value, "detail": detail})
        raise RuntimeError_(code, detail)

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------
    def create_session(self) -> str:
        if self.state() not in (RuntimeState.READY, RuntimeState.GENERATING):
            raise RuntimeError_(ErrorCode.RUNTIME_START_FAILED,
                                  f"cannot create session in state {self.state().value}")
        with self._sessions_lock:
            if len(self._sessions) >= MAX_CONCURRENT_SESSIONS:
                raise RuntimeError_(ErrorCode.RUNTIME_START_FAILED,
                                      f"too many sessions (>{MAX_CONCURRENT_SESSIONS})")
            sid = str(uuid.uuid4())
            self._sessions[sid] = SessionState(session_id=sid)
        self._emit(EventKind.SESSION_CREATED, session_id=sid)
        return sid

    def reset_session(self, session_id: str) -> None:
        with self._sessions_lock:
            if session_id not in self._sessions:
                raise RuntimeError_(ErrorCode.SESSION_NOT_FOUND, session_id)
            self._sessions[session_id].token_history.clear()
            self._sessions[session_id].last_used_at = time.time()
        self._emit(EventKind.SESSION_RESET, session_id=session_id)

    def close_session(self, session_id: str) -> None:
        with self._sessions_lock:
            if session_id not in self._sessions:
                raise RuntimeError_(ErrorCode.SESSION_NOT_FOUND, session_id)
            self._sessions[session_id].closed = True
            del self._sessions[session_id]
        self._emit(EventKind.SESSION_CLOSED, session_id=session_id)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def submit_prompt(self, session_id: str, prompt: str,
                          options: Optional[GenerationOptions] = None) -> str:
        """Submit a prompt for streaming generation. Returns request_id.
        The generation runs on a background thread; events stream through
        the handler."""
        if self.state() != RuntimeState.READY:
            raise RuntimeError_(ErrorCode.REQUEST_ALREADY_ACTIVE,
                                  f"runtime state={self.state().value}")
        with self._sessions_lock:
            if session_id not in self._sessions:
                raise RuntimeError_(ErrorCode.SESSION_NOT_FOUND, session_id)
        opts = validate_settings(options or GenerationOptions(),
                                     vocab_size=self._tokenizer.vocab_size)
        if not isinstance(prompt, str):
            raise RuntimeError_(ErrorCode.PROMPT_EMPTY, "prompt must be str")
        if len(prompt) == 0:
            raise RuntimeError_(ErrorCode.PROMPT_EMPTY, "prompt empty")
        if len(prompt.encode("utf-8")) > MAX_PROMPT_BYTES:
            raise RuntimeError_(ErrorCode.PROMPT_TOO_LARGE,
                                  f">{MAX_PROMPT_BYTES} bytes")
        if len(prompt) > MAX_PROMPT_CHARS:
            raise RuntimeError_(ErrorCode.PROMPT_TOO_LARGE,
                                  f">{MAX_PROMPT_CHARS} chars")

        rid = str(uuid.uuid4())
        self._active_request_id = rid
        self._cancel_event.clear()
        self._transition(RuntimeState.GENERATING)
        self._emit(EventKind.GENERATION_QUEUED, session_id=session_id, request_id=rid,
                     payload={"prompt_chars": len(prompt)})
        t = threading.Thread(target=self._generate,
                                 args=(session_id, prompt, opts, rid),
                                 daemon=True)
        self._active_generation_thread = t
        t.start()
        return rid

    def submit_prompt_sync(self, session_id: str, prompt: str,
                                options: Optional[GenerationOptions] = None) -> Dict[str, Any]:
        """Testing helper — runs generation synchronously; returns the
        collected result dict when done."""
        collected: List[Dict[str, Any]] = []
        prior_handler = self._event_handler

        def _chain(ev: RuntimeEvent):
            collected.append(ev.to_dict())
            if prior_handler is not None:
                prior_handler(ev)
        self._event_handler = _chain
        try:
            rid = self.submit_prompt(session_id, prompt, options)
            self._active_generation_thread.join(timeout=300)
        finally:
            self._event_handler = prior_handler
        return {"request_id": rid, "events": collected}

    def cancel(self, request_id: str) -> bool:
        if self._active_request_id != request_id:
            return False
        self._cancel_event.set()
        self._emit(EventKind.GENERATION_CANCEL_REQUESTED,
                     request_id=request_id, payload={})
        with self._state_lock:
            if self._state == RuntimeState.GENERATING:
                self._state = RuntimeState.CANCELLING
        return True

    def _apply_repetition_penalty(self, logits: torch.Tensor,
                                        history: List[int], penalty: float) -> torch.Tensor:
        if penalty == 1.0 or not history:
            return logits
        for tok in set(history):
            if 0 <= tok < logits.size(-1):
                v = logits[..., tok]
                logits[..., tok] = torch.where(v < 0, v * penalty, v / penalty)
        return logits

    def _sample_token(self, logits: torch.Tensor, opts: GenerationOptions,
                          rng: torch.Generator) -> int:
        # Deterministic greedy at temperature 0
        if opts.temperature == 0.0:
            return int(logits.argmax(dim=-1).item())
        scaled = logits / max(opts.temperature, 1e-6)
        # top-k
        if opts.top_k > 0:
            top_vals, top_idx = torch.topk(scaled, min(opts.top_k, scaled.size(-1)))
            mask = torch.full_like(scaled, float("-inf"))
            mask.scatter_(-1, top_idx, top_vals)
            scaled = mask
        # top-p
        if opts.top_p < 1.0:
            sorted_vals, sorted_idx = torch.sort(scaled, descending=True)
            probs = torch.softmax(sorted_vals, dim=-1)
            cum = probs.cumsum(dim=-1)
            cut = cum > opts.top_p
            # Shift right to always include at least one token
            cut[..., 1:] = cut[..., :-1].clone()
            cut[..., 0] = False
            sorted_vals[cut] = float("-inf")
            scaled = torch.full_like(scaled, float("-inf"))
            scaled.scatter_(-1, sorted_idx, sorted_vals)
        probs = torch.softmax(scaled, dim=-1)
        # sample
        return int(torch.multinomial(probs, 1, generator=rng).item())

    def _generate(self, session_id: str, prompt: str,
                     opts: GenerationOptions, request_id: str) -> None:
        try:
            with self._sessions_lock:
                sess = self._sessions.get(session_id)
                if sess is None or sess.closed:
                    raise RuntimeError_(ErrorCode.SESSION_NOT_FOUND, session_id)

            tok = self._tokenizer
            model = self._model
            prompt_ids = tok.encode(prompt, add_bos=False, add_eos=False)
            if len(prompt_ids) > MAX_PROMPT_TOKENS:
                raise RuntimeError_(ErrorCode.PROMPT_TOO_LARGE,
                                      f">{MAX_PROMPT_TOKENS} tokens")
            # Policy B: deterministic replay of session history + this prompt
            with self._sessions_lock:
                context = list(sess.token_history) + prompt_ids
            if len(context) > MAX_SESSION_TOKENS:
                # Deterministic tail truncation at token boundaries.
                context = context[-MAX_SESSION_TOKENS:]
                self._emit(EventKind.RUNTIME_WARNING,
                             session_id=session_id, request_id=request_id,
                             payload={"warning": "prompt_context_truncated",
                                        "kept_tokens": MAX_SESSION_TOKENS})

            self._emit(EventKind.GENERATION_STARTED,
                         session_id=session_id, request_id=request_id,
                         payload={"prompt_tokens": len(prompt_ids),
                                    "context_tokens": len(context)})

            rng = torch.Generator()
            if opts.deterministic_seed is not None:
                rng.manual_seed(int(opts.deterministic_seed))
            else:
                rng.manual_seed(0xA0EA)

            generated_ids: List[int] = []
            # Renderer correction (EN-TRAIN §21 / spec §21).
            # Previously, TEXT_DELTA payloads were computed by
            # per-token decode `tok.decode(generated_ids[-1:])`, which
            # drops the SentencePiece leading-space marker so word
            # boundaries disappear in the UI. The corrected renderer
            # decodes the CUMULATIVE token sequence and emits the
            # tail-only delta, so concatenating every emitted delta
            # exactly reproduces the canonical one-shot decode:
            #   D_stream(y) = D_full(y).
            #
            # Byte-fallback safety: multi-byte UTF-8 characters can be
            # split across tokens by the byte-fallback path. SentencePiece
            # returns U+FFFD (REPLACEMENT CHARACTER) for an incomplete
            # UTF-8 tail. We must not commit that mojibake — it would
            # be silently replaced by the real character on the next
            # token, and a length-based delta would miss the swap.
            # Fix: strip trailing U+FFFD from the cumulative decode
            # before computing the delta. Any incomplete tail is HELD
            # BACK until it completes into a real code point. At EOS,
            # the final full_text uses the same canonical one-shot
            # decode as the loss / evaluation path.
            # No change to model weights, token selection, logits,
            # generation order, tokenizer files, model configuration,
            # or the A0 architecture fingerprint. See
            # docs/en_train/RENDERER_FIX_PROOF.md.
            emitted_text = ""        # what D_stream has emitted so far
            t_first = None
            t0 = time.time()
            for step in range(opts.max_new_tokens):
                if self._cancel_event.is_set():
                    self._emit(EventKind.GENERATION_CANCELLED,
                                 session_id=session_id, request_id=request_id,
                                 payload={"generated_tokens": len(generated_ids),
                                            "committed_text": tok.decode(generated_ids)})
                    with self._state_lock:
                        self._state = RuntimeState.READY
                    self._active_request_id = None
                    return
                ids = torch.tensor([context + generated_ids], dtype=torch.long)
                with torch.inference_mode():
                    out = model(input_ids=ids)
                logits = out.logits[0, -1, :].clone()
                logits = self._apply_repetition_penalty(
                    logits, context + generated_ids, opts.repetition_penalty)
                next_tok = self._sample_token(logits, opts, rng)
                generated_ids.append(next_tok)
                if t_first is None:
                    t_first = time.time() - t0
                # Canonical-decode-and-delta renderer (§21). Decode
                # the entire generated sequence and emit only the
                # newly-appended tail. Strip trailing U+FFFD to hold
                # back incomplete UTF-8 tails until the multi-byte
                # sequence completes. On EOS / natural end, the
                # completion event carries the full canonical decode
                # so any remaining U+FFFD is visible to the caller
                # (matching what a one-shot decode would produce).
                canonical_so_far = tok.decode(generated_ids)
                committable = canonical_so_far.rstrip("�")
                if committable.startswith(emitted_text):
                    text_delta = committable[len(emitted_text):]
                    emitted_text = committable
                else:
                    # Prior emission is no longer a prefix — replacement
                    # of a mojibake by a valid character mid-string.
                    # Emit nothing this step; the committable tail will
                    # be reconciled on the next step or at completion.
                    text_delta = ""
                self._emit(EventKind.TOKEN_GENERATED,
                             session_id=session_id, request_id=request_id,
                             payload={"token_id": next_tok,
                                        "step": step + 1})
                self._emit(EventKind.TEXT_DELTA,
                             session_id=session_id, request_id=request_id,
                             payload={"delta": text_delta})
                # EOS
                if next_tok == tok.eos_id:
                    break
            wall_s = time.time() - t0
            tps = len(generated_ids) / wall_s if wall_s > 0 else 0
            # Final flush of the delta stream. If a byte-fallback
            # multi-byte tail was still incomplete on the last step
            # (or if the U+FFFD-hold logic held back a legitimate
            # closing character), emit the missing suffix now so
            # that the concatenation of every TEXT_DELTA equals
            # tok.decode(generated_ids) exactly.
            _final_full = tok.decode(generated_ids)
            if _final_full != emitted_text:
                _tail = _final_full[len(emitted_text):] if _final_full.startswith(emitted_text) else _final_full
                self._emit(EventKind.TEXT_DELTA,
                             session_id=session_id, request_id=request_id,
                             payload={"delta": _tail, "flush": True})
                emitted_text = _final_full
            # Commit to session
            with self._sessions_lock:
                sess.token_history.extend(prompt_ids)
                sess.token_history.extend(generated_ids)
                sess.last_used_at = time.time()
            self._last_gen_stats = {
                "generated_tokens": len(generated_ids),
                "wall_seconds": wall_s,
                "tokens_per_second": tps,
                "first_token_latency_seconds": t_first,
            }
            self._emit(EventKind.GENERATION_COMPLETED,
                         session_id=session_id, request_id=request_id,
                         payload={"generated_tokens": len(generated_ids),
                                    "wall_seconds": wall_s,
                                    "tokens_per_second": tps,
                                    "first_token_latency_seconds": t_first,
                                    "full_text": tok.decode(generated_ids)})
            with self._state_lock:
                self._state = RuntimeState.READY
            self._active_request_id = None
        except RuntimeError_ as e:
            self._emit(EventKind.GENERATION_FAILED,
                         session_id=session_id, request_id=request_id,
                         payload={"code": e.code.value, "detail": e.detail})
            with self._state_lock:
                self._state = RuntimeState.READY
            self._active_request_id = None
        except Exception as e:
            self._emit(EventKind.GENERATION_FAILED,
                         session_id=session_id, request_id=request_id,
                         payload={"code": ErrorCode.GENERATION_FAILED.value,
                                    "detail": str(e)[:200]})
            with self._state_lock:
                self._state = RuntimeState.READY
            self._active_request_id = None

    # ------------------------------------------------------------------
    def diagnostics(self) -> Dict[str, Any]:
        return {
            "desktop_runtime_version": "aeon-desktop-runtime-1.0.0",
            "state": self.state().value,
            "release_id": (self._release_manifest or {}).get("release_id"),
            "release_label": (self._release_manifest or {}).get("release_label"),
            "parameter_count": (self._release_manifest or {}).get("parameter_count"),
            "K": 16,
            "ACIS_default": "OFF",
            "session_count": len(self._sessions),
            "active_request_id": self._active_request_id,
            "last_generation": dict(self._last_gen_stats),
            "torch_threads": torch.get_num_threads(),
            "network_policy": "offline_only",
        }

    # ------------------------------------------------------------------
    def shutdown(self, timeout_s: float = 30.0) -> None:
        with self._state_lock:
            if self._state == RuntimeState.STOPPED:
                return
            prev = self._state
            self._state = RuntimeState.SHUTTING_DOWN
        self._emit(EventKind.RUNTIME_SHUTDOWN_STARTED, payload={"prev_state": prev.value})
        if self._active_request_id is not None:
            self._cancel_event.set()
        t = self._active_generation_thread
        if t is not None and t.is_alive():
            t.join(timeout=timeout_s)
        # Release sessions
        with self._sessions_lock:
            self._sessions.clear()
        # Drop model + tokenizer refs
        self._model = None
        self._tokenizer = None
        with self._state_lock:
            self._state = RuntimeState.STOPPED
        self._emit(EventKind.RUNTIME_SHUTDOWN_COMPLETED)
