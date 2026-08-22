"""aeon.en_train.proof_harness — ENGLISH-PROOF-0 / A helpers.

Two utilities the pilot uses to produce weight-only attribution
evidence. They contain no training or online-update logic.

  * ``AttributionSettings`` — the frozen decoding + serialization
    contract every checkpoint MUST run under so any observed
    difference is attributable to θ only.

  * ``run_attribution`` — given a list of prompts and two
    checkpoints (parent P2 and candidate), executes generation for
    each prompt under identical settings and returns a list of
    per-response records suitable for direct JSONL logging. It
    performs no rewriting, no post-generation grammar repair, no
    external-model calls, and no logit or token modification.

The harness does not update θ.
"""
from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class AttributionSettings:
    """Frozen decoding + serialization contract for weight-only
    attribution. Every checkpoint compared under a proof run must
    use bytewise-identical values here — the only permitted
    difference between two runs is the checkpoint bytes.
    """
    context_length: int = 2048
    max_new_tokens: int = 256
    greedy: bool = True                      # temperature=0
    temperature: float = 0.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    stop_on_eos: bool = True
    deterministic_seed: int = 20260822
    prompt_serialization_id: str = "aeon_desktop_runtime_v1"
    renderer_id: str = "aeon_desktop_runtime_v1_streamed_canonical"

    def fingerprint(self) -> str:
        d = asdict(self)
        canon = "|".join(f"{k}={d[k]}" for k in sorted(d.keys()))
        return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


@dataclass
class AttributionResponse:
    prompt_id: str
    prompt_text: str
    checkpoint_role: str                    # "parent_P2" or "candidate"
    checkpoint_sha256: str
    generated_token_ids: List[int]
    per_step_selected_token: List[int]
    full_decoded_text: str
    streamed_decoded_text: str
    stop_reason: str
    generation_settings_fingerprint: str
    generation_duration_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _sha256_of_path(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _sha256_of_state_dict(sd: Dict[str, Any]) -> str:
    """Deterministic hash of a state_dict for logging."""
    import torch  # local import — the harness itself is torch-free
    h = hashlib.sha256()
    for k in sorted(sd.keys()):
        t = sd[k].detach().to(torch.float32).cpu().contiguous()
        h.update(k.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return "sha256:" + h.hexdigest()


def stream_and_full_decode(tokenizer, ids: List[int]) -> Tuple[str, str]:
    """Reference implementation of the runtime's cumulative decode +
    U+FFFD hold-back + completion-time flush. Returns (streamed, full).
    Used by the attribution harness to record BOTH strings so the
    proof evaluation can assert D_stream == D_full without depending
    on the runtime's event stream.
    """
    if not ids:
        return "", ""
    D_full = tokenizer.decode(ids)
    emitted = ""
    deltas: List[str] = []
    for i in range(1, len(ids) + 1):
        canonical_so_far = tokenizer.decode(ids[:i])
        committable = canonical_so_far.rstrip("�")
        if committable.startswith(emitted):
            deltas.append(committable[len(emitted):])
            emitted = committable
    if emitted != D_full:
        tail = D_full[len(emitted):] if D_full.startswith(emitted) else D_full
        deltas.append(tail)
    return "".join(deltas), D_full


def run_attribution(*, tokenizer, model, prompts: List[Tuple[str, str]],
                    settings: AttributionSettings,
                    checkpoint_role: str,
                    checkpoint_sha256: str,
                    forward_step_fn: Callable[[Any, List[int]], int],
                    ) -> List[AttributionResponse]:
    """Execute one attribution pass.

    prompts: list of (prompt_id, prompt_text) tuples.
    forward_step_fn: a callable that takes (model, context_ids) and
      returns the next-token id under the frozen greedy decoding rule.
      Kept as an injected dependency so tests can drive the harness
      with a deterministic stub without loading a real model.

    Returns a list of AttributionResponse; makes no writes and does
    not mutate model parameters.
    """
    fp = settings.fingerprint()
    out: List[AttributionResponse] = []
    for prompt_id, prompt_text in prompts:
        t0 = time.time()
        context = tokenizer.encode(prompt_text, add_bos=False, add_eos=False)
        generated: List[int] = []
        per_step: List[int] = []
        stop_reason = "max_new_tokens"
        for _ in range(settings.max_new_tokens):
            nxt = forward_step_fn(model, list(context) + list(generated))
            generated.append(int(nxt))
            per_step.append(int(nxt))
            if settings.stop_on_eos and int(nxt) == int(tokenizer.eos_id):
                stop_reason = "eos"
                break
        streamed, full = stream_and_full_decode(tokenizer, generated)
        out.append(AttributionResponse(
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            checkpoint_role=checkpoint_role,
            checkpoint_sha256=checkpoint_sha256,
            generated_token_ids=list(generated),
            per_step_selected_token=list(per_step),
            full_decoded_text=full,
            streamed_decoded_text=streamed,
            stop_reason=stop_reason,
            generation_settings_fingerprint=fp,
            generation_duration_seconds=time.time() - t0,
        ))
    return out


def assert_attribution_settings_bytewise_equal(a: AttributionSettings,
                                               b: AttributionSettings) -> None:
    """Fail loudly if any field differs; the only permitted difference
    between two attribution runs is θ."""
    if a.fingerprint() != b.fingerprint():
        raise RuntimeError(
            f"attribution settings differ:\n  a={a}\n  b={b}\n"
            f"  fp(a)={a.fingerprint()}\n  fp(b)={b.fingerprint()}")
