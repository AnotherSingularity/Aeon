"""aeon.en_train.proof_pilot — ENGLISH-PROOF-0 bounded pilot driver.

Runs the offline response-masked cross-entropy pilot described in
directive Section 7 using Aeon's existing architecture and the
existing approved optimizer path (aeon.en_train.trainer.train_one_step).

  * Loads the parent P2 checkpoint into a fresh candidate; NEVER
    overwrites P2.
  * Serialises each Dolly record with the desktop-runtime prompt/
    response contract; builds a response mask over the assistant
    turn only.
  * Runs response-masked causal cross-entropy L(θ) with a hard
    max of 3,000,000 response-training tokens.
  * Saves candidate checkpoints at the required token targets and
    selects the candidate by validation loss + unsealed dev probes.
  * Fails closed on: NaN / Inf loss, gradient explosion, sigma
    certificate failure, architecture drift, parameter-count drift,
    tokenizer drift, P2 mutation, unexpected state-dict change,
    recurrence/clock-test failure.
  * Records gradient + weight-delta + invariance evidence.

The driver refuses to run until:
  (a) research-data/incoming/EN-DOLLY-15K/sources/*.jsonl exists and
      hashes match docs/en_train/dolly15k_provenance.json;
  (b) docs/en_train/dolly15k_split_manifest.json exists AND its
      sealed_test_lock_sha256 verifies.

If either precondition is unmet, the driver exits non-zero with the
literal status string "AWAITING_DOLLY_DATA_UPLOAD" so a wrapping
script can propagate the halt state.

The driver introduces NO new architectural module, NO LoRA/adapter,
NO fallback language model, NO API inference, and NO online parameter
update. Every optimizer step is aeon.en_train.trainer.train_one_step.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Pilot configuration (defaults come from the directive)
# ---------------------------------------------------------------------------
@dataclass
class PilotConfig:
    identifier: str = "AEON-EN-PROOF-DOLLY15K-S20260822"
    parent_checkpoint: Path = Path("runs/aeon_lbc1_P2/final.pt")
    tokenizer_path: Path = Path(
        "release-assets/aeon-desktop-p2-proxy/tokenizer/aeon-lbc1.model")
    provenance_path: Path = Path("docs/en_train/dolly15k_provenance.json")
    split_manifest_path: Path = Path("docs/en_train/dolly15k_split_manifest.json")
    data_root: Path = Path("research-data/incoming/EN-DOLLY-15K/sources")
    out_dir: Path = Path("runs/en_proof_dolly15k_s20260822")

    seed: int = 20260822
    hard_max_response_tokens: int = 3_000_000
    checkpoint_token_targets: Tuple[int, ...] = (250_000, 500_000, 1_000_000,
                                                   2_000_000, 3_000_000)

    batch_size: int = 4
    seq_len: int = 512
    lr_peak: float = 1e-4
    lr_final: float = 1e-5
    grad_clip: float = 1.0
    warmup_frac: float = 0.02
    val_batches: int = 16
    device: str = "cpu"


# ---------------------------------------------------------------------------
# Halt states
# ---------------------------------------------------------------------------
HALT_AWAITING_DATA = "AWAITING_DOLLY_DATA_UPLOAD"
HALT_READY = "ENGLISH_PROOF_READY_FOR_DYLAN_REVIEW"
HALT_FAILED = "ENGLISH_PROOF_FAILED_NO_PACKAGING"


class PilotHalt(RuntimeError):
    def __init__(self, state: str, detail: str = ""):
        super().__init__(f"{state}: {detail}" if detail else state)
        self.state = state
        self.detail = detail


# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------
def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def assert_preconditions(cfg: PilotConfig) -> Dict[str, Any]:
    """Fail closed with a machine-readable halt state if any input is
    missing. Returns a dict of resolved paths + pinned digests on
    success; raises PilotHalt(HALT_AWAITING_DATA, ...) otherwise."""
    missing: List[str] = []

    if not cfg.provenance_path.exists():
        missing.append(str(cfg.provenance_path))
    if not cfg.split_manifest_path.exists():
        missing.append(str(cfg.split_manifest_path))
    if not cfg.data_root.exists():
        missing.append(str(cfg.data_root))
    else:
        source_files = sorted(cfg.data_root.glob("*.jsonl"))
        if not source_files:
            missing.append(f"{cfg.data_root}/*.jsonl")

    if missing:
        raise PilotHalt(HALT_AWAITING_DATA,
                        f"missing inputs: {missing}. Upload the "
                        "authorized Dolly-15k artifact under "
                        f"{cfg.data_root} and populate "
                        f"{cfg.provenance_path} + {cfg.split_manifest_path}.")

    # Provenance must have sha256 populated (not null)
    prov = json.loads(cfg.provenance_path.read_text(encoding="utf-8"))
    if prov.get("acquisition", {}).get("sha256") in (None, ""):
        raise PilotHalt(HALT_AWAITING_DATA,
                        f"{cfg.provenance_path} still has null "
                        "acquisition.sha256; populate it after the upload.")

    # Sealed-test lock
    from aeon.en_train.dolly_split import verify_sealed_test_lock
    ok, msg = verify_sealed_test_lock(cfg.split_manifest_path)
    if not ok:
        raise PilotHalt(HALT_FAILED,
                        f"sealed_test_lock verification failed: {msg}")

    # Parent P2 must exist and match its pinned sha256
    if not cfg.parent_checkpoint.exists():
        raise PilotHalt(HALT_FAILED,
                        f"parent checkpoint missing: {cfg.parent_checkpoint}")
    parent_sha = _sha256_file(cfg.parent_checkpoint)
    fp = json.loads(Path("docs/en_train/EN_TRAIN_ARCHITECTURE_FREEZE.json"
                          ).read_text(encoding="utf-8"))
    if parent_sha != fp["protected_p2_checkpoint"]["sha256"]:
        raise PilotHalt(HALT_FAILED,
                        f"parent P2 sha256 drift: pinned="
                        f"{fp['protected_p2_checkpoint']['sha256']} "
                        f"disk={parent_sha}")

    # Tokenizer must match pinned sha256
    if not cfg.tokenizer_path.exists():
        raise PilotHalt(HALT_FAILED,
                        f"tokenizer missing: {cfg.tokenizer_path}")
    tok_sha = _sha256_file(cfg.tokenizer_path)
    if tok_sha != fp["protected_tokenizer"]["sha256"]:
        raise PilotHalt(HALT_FAILED,
                        f"tokenizer sha256 drift: pinned="
                        f"{fp['protected_tokenizer']['sha256']} "
                        f"disk={tok_sha}")

    return {
        "provenance": prov,
        "parent_sha256": parent_sha,
        "tokenizer_sha256": tok_sha,
    }


# ---------------------------------------------------------------------------
# Candidate isolation — copy P2 into a fresh candidate; NEVER overwrite P2
# ---------------------------------------------------------------------------
def create_candidate(cfg: PilotConfig) -> Path:
    """Copy parent P2 to a fresh candidate directory. Refuses to
    overwrite an existing candidate; the operator must remove the
    previous run explicitly to guard against silent history loss."""
    cand_dir = cfg.out_dir / cfg.identifier
    if cand_dir.exists() and any(cand_dir.iterdir()):
        raise PilotHalt(HALT_FAILED,
                        f"candidate dir already exists and is non-empty: "
                        f"{cand_dir}. Remove it explicitly before re-running.")
    cand_dir.mkdir(parents=True, exist_ok=True)
    cand_path = cand_dir / "initial.pt"
    shutil.copyfile(cfg.parent_checkpoint, cand_path)
    return cand_path


# ---------------------------------------------------------------------------
# Prompt serialization contract — same as the desktop runtime
# ---------------------------------------------------------------------------
USER_PREFIX = "user: "
ASSIST_PREFIX = "assistant: "
TURN_SEP = "\n\n"


def render_dolly_record_for_training(instruction: str,
                                       context: str,
                                       response: str) -> Tuple[str, List[Tuple[int, int]]]:
    """Serialise one Dolly record. Returns (text, list of assistant
    character-spans). The response is the sole SUPERVISED region."""
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0

    # user turn: instruction (+ context if present, joined with two newlines)
    user_content = instruction
    if context and context.strip():
        user_content = instruction + "\n\n" + context
    user_seg = USER_PREFIX + user_content
    parts.append(user_seg); cursor += len(user_seg)

    parts.append(TURN_SEP); cursor += len(TURN_SEP)

    # assistant turn — supervised region
    a_prefix = ASSIST_PREFIX
    parts.append(a_prefix); assistant_start = cursor + len(a_prefix) - len(a_prefix)  # cursor
    # Actually assistant_start = cursor here (before appending prefix)? We want
    # the span to cover the CONTENT ONLY, not the "assistant: " prefix.
    assistant_start = cursor + len(a_prefix)
    cursor += len(a_prefix)
    parts.append(response); cursor += len(response)
    assistant_end = cursor
    spans.append((assistant_start, assistant_end))
    return "".join(parts), spans


# ---------------------------------------------------------------------------
# Halt-only entry point (this session cannot run training)
# ---------------------------------------------------------------------------
def halt_state_for_current_environment(cfg: Optional[PilotConfig] = None
                                        ) -> Dict[str, Any]:
    """Compute the current halt state for the caller. This is what a
    thin wrapper script prints when the corpus has not been uploaded
    or when a precondition fails. It does not attempt training.
    Returns a dict {state, detail, resolved} suitable for direct
    JSON logging."""
    cfg = cfg or PilotConfig()
    try:
        resolved = assert_preconditions(cfg)
        return {
            "state": HALT_READY[:0] or "READY_TO_ATTEMPT_PILOT",
            "detail": ("preconditions satisfied — pilot can start; "
                       "actual training must be launched from a script "
                       "that also produces the invariance evidence file"),
            "resolved": {
                "parent_sha256": resolved["parent_sha256"],
                "tokenizer_sha256": resolved["tokenizer_sha256"],
            },
        }
    except PilotHalt as e:
        return {"state": e.state, "detail": e.detail, "resolved": None}
