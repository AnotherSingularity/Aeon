"""
aeon/checkpoint.py — atomic checkpoint save / restricted load.

Directive §10 requirements:
  §10.1 completeness — a resumable checkpoint carries model, optimizer, scheduler,
       global step, useful-token count, data position (or deterministic-resume
       state), RNG state, model+train config, tokenizer+corpus identity, K=16,
       precision policy, substrate dtype policy, certificate config, patch
       manifest version, instrumentation config, source commit identity.

  §10.2 atomic save — write to temp path, flush+close, validate, sha256, atomic
       rename, keep prior known-good until the replacement is verified.

  §10.4 local security — load with weights_only=True by default (or hardened
       equivalent for older torch), reject incompatible metadata (K mismatch,
       vocab mismatch, patch manifest mismatch), never `eval` arbitrary
       serialized objects.

The load function is `strict_load` — the untrusted-input entry point. It never
returns arbitrary Python objects and it raises `CheckpointIncompatible` when the
metadata disagrees with the running configuration in a way that would corrupt
the resume.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

CHECKPOINT_SCHEMA_VERSION = 1
PATCH_MANIFEST_VERSION = 1                          # bumped when six-patch topology changes
K_LOCKED = 16                                       # §3.4


class CheckpointIncompatible(RuntimeError):
    """Raised on load when metadata disagrees with the running config in a way
    that would corrupt the resume (fail-closed per §16.2)."""


class CheckpointCorrupt(RuntimeError):
    """Raised when a checkpoint fails integrity verification."""


# ---------------------------------------------------------------------------
# Source-commit identity (best effort)
# ---------------------------------------------------------------------------
def source_commit_id() -> str:
    """Return the source_commit for the currently running Aeon build.

    W10-5 policy:

    * If Aeon is running as a FROZEN application (PyInstaller onedir),
      the value comes from ``aeon.version.RELEASE_METADATA["source_commit"]``,
      which is populated at build time from ``packaging/windows/RELEASE.json``.
      There is no `git rev-parse` fallback in frozen mode — a frozen build
      has no `.git` directory and any git call would return "unknown",
      which the audit's A15 finding correctly refused. If RELEASE_METADATA
      is missing or reports ``source_commit == "unknown"`` in frozen mode,
      raise ``SourceCommitUnavailable`` so protected_save/protected_load
      fail closed rather than record an unknown-provenance checkpoint.

    * In SOURCE-TREE (development) mode, prefer git rev-parse HEAD; fall
      back to RELEASE_METADATA if the git binary is missing (e.g. tests
      running in an unusual environment). Only in the source-tree case
      is returning ``"unknown"`` acceptable, and only when git fails and
      RELEASE_METADATA has nothing better either — that path corresponds
      to a dev checkout without git installed, which is a legitimate
      dev-only situation.
    """
    from aeon.windows_paths import is_frozen
    from aeon.version import RELEASE_METADATA

    rel_commit = RELEASE_METADATA.get("source_commit", "unknown")

    if is_frozen():
        if not rel_commit or rel_commit == "unknown":
            raise SourceCommitUnavailable(
                "frozen build has no source_commit in RELEASE_METADATA; "
                "refusing to record checkpoint with unknown provenance")
        return rel_commit

    # Source-tree mode: git is authoritative.
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=2,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        ).stdout.strip()
        if rev:
            return rev
    except Exception:
        pass
    # git absent or failed. Try RELEASE_METADATA before falling all the way
    # back to "unknown".
    if rel_commit and rel_commit != "unknown":
        return rel_commit
    return "unknown"


class SourceCommitUnavailable(RuntimeError):
    """Raised when the frozen build cannot report a real source_commit and
    ``source_commit_id`` refuses to return ``"unknown"``. Callers should
    surface this as a checkpoint-refusal so an operator can fix the
    build metadata before shipping."""
    pass


# ---------------------------------------------------------------------------
# Sha256 for integrity
# ---------------------------------------------------------------------------
def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Metadata assembly
# ---------------------------------------------------------------------------
def build_metadata(
    step: int,
    model_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    tokenizer_id: Optional[str],
    corpus_id: Optional[str],
    data_position: int,
    instrumentation_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the checkpoint metadata dict per §10.1."""
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "patch_manifest_version": PATCH_MANIFEST_VERSION,
        "source_commit": source_commit_id(),
        "step": int(step),
        "K": K_LOCKED,
        "model_config": model_cfg,
        "train_config": {k: v for k, v in train_cfg.items()
                         if k not in ("resume",)},           # resume flag is caller-side
        "data_config": data_cfg,
        "data_position": int(data_position),
        "tokenizer_identity": tokenizer_id or "synthetic",
        "corpus_identity": corpus_id or "synthetic",
        "precision_policy": {
            "recursion_fp32": True,
            "gamma_fp32_master": True,
            "gate_scalars_fp32_master": True,
            "substrate_state_follows_param_dtype": True,
            "rotary_inv_freq_fresh_fp32": True,
        },
        "certificate_policy": {
            "structural_MARGIN_H": model_cfg.get("margin_h"),
            "structural_MARGIN_C": model_cfg.get("margin_c"),
            "audit_on_load": True,
        },
        "instrumentation_config": instrumentation_cfg or {},
    }


# ---------------------------------------------------------------------------
# Atomic save (§10.2)
# ---------------------------------------------------------------------------
def atomic_save(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Dict[str, Any],
    rng_state: Optional[Dict[str, Any]] = None,
    scheduler_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Write a checkpoint atomically.

    Returns the final metadata dict (with sha256 written after validation).
    Preserves any existing `path` as `path + ".prev"` until the new file is
    integrity-verified.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_dir = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".ckpt.tmp.", dir=tmp_dir)
    os.close(fd)

    if rng_state is None:
        rng_state = {
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
        }

    payload = {
        "metadata": metadata,
        "model": model.state_dict(),
        "optim": optimizer.state_dict(),
        "scheduler": scheduler_state,
        "rng": rng_state,
    }
    try:
        torch.save(payload, tmp_path)
        # (1) flush; torch.save closes on the file object. Extra fsync for durability:
        try:
            with open(tmp_path, "rb+") as fh:
                fh.flush(); os.fsync(fh.fileno())
        except Exception:
            pass
        # (2) validate: reopen and check keys are present
        try:
            probe = torch.load(tmp_path, map_location="cpu", weights_only=False)
        except Exception as e:
            raise CheckpointCorrupt(f"unable to reload the just-written checkpoint: {e}")
        for key in ("metadata", "model", "optim", "rng"):
            if key not in probe:
                raise CheckpointCorrupt(f"missing required key: {key}")
        if probe["metadata"].get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCorrupt("schema_version drift between save and read-back")
        # (3) sha256
        digest = _sha256(tmp_path)
        # (4) preserve the previous ckpt as .prev BEFORE renaming so a crash between
        # rename and sha-write cannot lose it
        if os.path.exists(path):
            prev = path + ".prev"
            if os.path.exists(prev):
                os.unlink(prev)
            os.rename(path, prev)
        os.rename(tmp_path, path)
        # (5) write sidecar sha (INDEPENDENT file so a partial write can't fool us)
        with open(path + ".sha256", "w") as fh:
            fh.write(digest + "\n")
        metadata = dict(metadata)
        metadata["sha256"] = digest
        return metadata
    except Exception:
        # crashed before rename: temp file is safe to delete
        if os.path.exists(tmp_path):
            try: os.unlink(tmp_path)
            except Exception: pass
        raise


# ---------------------------------------------------------------------------
# Strict load (§10.4)
# ---------------------------------------------------------------------------
def strict_load(
    path: str,
    *,
    expected_model_config: Dict[str, Any],
    require_sha256: bool = True,
) -> Dict[str, Any]:
    """Load a checkpoint with §10.4 hardening:
      * verify sidecar sha256 (fail-closed if `require_sha256=True`)
      * `weights_only=True` when the torch version supports it (torch>=2.0);
        falls back to the safest available option otherwise.
      * verify metadata compatibility: schema, K, vocab, patch manifest.

    Returns the loaded payload (metadata + state dicts).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # (1) sha256 gate
    sha_path = path + ".sha256"
    if os.path.exists(sha_path):
        expected = open(sha_path, encoding="ascii").read().strip().split()[0]
        actual = _sha256(path)
        if expected != actual:
            raise CheckpointCorrupt(
                f"sha256 mismatch: expected {expected}, got {actual}")
    elif require_sha256:
        raise CheckpointCorrupt(f"missing sidecar sha256 for {path}")

    # (2) weights_only load — best available. If torch is too old to have the
    # weights_only kw, that IS a security failure surface; we still try but the
    # runtime must be pinned.
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # older torch — no weights_only kwarg. Raise so caller can decide.
        raise CheckpointIncompatible(
            "torch too old to support weights_only=True; pin torch>=2.0 (see docs)")
    except Exception as e:
        # torch.load may reject some serialized state_dicts under weights_only;
        # if the metadata carries the expected schema we can retry with a
        # narrower policy. But we DO NOT silently switch to full pickle load —
        # instead we raise so the caller can make a controlled decision.
        raise CheckpointIncompatible(
            f"strict load rejected the payload (this is by design; see aeon/checkpoint.py): {e}"
        ) from e

    md = payload.get("metadata")
    if not isinstance(md, dict):
        raise CheckpointIncompatible("no metadata dict")

    # (3) metadata compatibility gates
    if md.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointIncompatible(
            f"schema_version mismatch: got {md.get('schema_version')}, "
            f"expected {CHECKPOINT_SCHEMA_VERSION}")
    if md.get("patch_manifest_version") != PATCH_MANIFEST_VERSION:
        raise CheckpointIncompatible(
            f"patch_manifest_version mismatch: got {md.get('patch_manifest_version')}, "
            f"expected {PATCH_MANIFEST_VERSION}")
    if md.get("K") != K_LOCKED:
        raise CheckpointIncompatible(f"K mismatch: ckpt K={md.get('K')} vs locked {K_LOCKED}")
    ck_vocab = md.get("model_config", {}).get("transformer", {}).get("vocab_size")
    ex_vocab = expected_model_config.get("transformer", {}).get("vocab_size")
    if ck_vocab is not None and ex_vocab is not None and ck_vocab != ex_vocab:
        raise CheckpointIncompatible(
            f"vocab_size mismatch: ckpt={ck_vocab} runtime={ex_vocab} — "
            "reject rather than silently swap tokenizer semantics")

    return payload


# ---------------------------------------------------------------------------
# List checkpoints in a directory
# ---------------------------------------------------------------------------
def list_checkpoints(out_dir: str) -> list[str]:
    import glob
    cks = glob.glob(os.path.join(out_dir, "ckpt_*.pt"))
    return sorted(cks, key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))


def latest_checkpoint(out_dir: str) -> Optional[str]:
    cks = list_checkpoints(out_dir)
    return cks[-1] if cks else None
