"""aeon/launcher/resume.py — authenticated checkpoint enumeration for W10-3.

The launcher's Resume and Recovery paths need a way to list the checkpoints
that are (a) present, (b) authenticated against the current job's HMAC key,
and (c) compatible with the current model configuration. This module owns
that enumeration and never mutates the checkpoints it inspects.

Every result carries enough identity to build audit events (checkpoint
path, step, authorized_step, mac_algo, tokenizer_id, corpus_id,
source_commit) without loading model or optimizer state — the actual load
happens inside the worker under ``protected_load``.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class CheckpointCandidate:
    """One enumerated authenticated checkpoint."""

    path: str
    step: int
    authorized_step: int
    tokenizer_id: Optional[str]
    corpus_id: Optional[str]
    source_commit: Optional[str]
    mac_algo: str
    envelope_version: int
    authenticated: bool
    reason: Optional[str] = None  # if authenticated=False, why


def _read_meta(ckpt_path: str) -> Optional[dict]:
    mp = ckpt_path + ".meta.json"
    if not os.path.exists(mp):
        return None
    try:
        return json.loads(open(mp, encoding="utf-8").read())
    except Exception:
        return None


def enumerate_checkpoints(
    checkpoint_dir: str,
    keyref,
) -> List[CheckpointCandidate]:
    """Return one CheckpointCandidate per ``ckpt_*.pt`` under
    ``checkpoint_dir`` (sorted by step, newest first). Each entry is
    authenticated against ``keyref`` using the same HMAC verification
    ``aeon.protected_checkpoint.protected_load`` performs, but WITHOUT
    loading the payload — cheap enough to run in the launcher on every
    Resume-menu open.

    Non-authenticated / malformed / pre-W10-2 checkpoints are still
    returned but with ``authenticated=False`` and a ``reason`` string so
    the launcher can display them (grayed out) rather than silently
    dropping them.
    """
    from aeon.protected_checkpoint import _hmac_over
    import hmac

    if not os.path.isdir(checkpoint_dir):
        return []
    pattern = os.path.join(checkpoint_dir, "ckpt_*.pt")
    candidates: List[CheckpointCandidate] = []
    for p in glob.glob(pattern):
        if p.endswith(".prev") or p.endswith(".tmp"):
            continue
        meta = _read_meta(p)
        if meta is None:
            candidates.append(CheckpointCandidate(
                path=p, step=-1, authorized_step=-1,
                tokenizer_id=None, corpus_id=None, source_commit=None,
                mac_algo="", envelope_version=0, authenticated=False,
                reason="no envelope metadata (.meta.json missing)"))
            continue
        inner = meta.get("inner_metadata") or {}
        mac_expected = meta.get("mac_hex")
        if not mac_expected:
            candidates.append(CheckpointCandidate(
                path=p, step=int(inner.get("step", -1)),
                authorized_step=int(meta.get("authorized_step", -1)),
                tokenizer_id=inner.get("tokenizer_id"),
                corpus_id=inner.get("corpus_id"),
                source_commit=meta.get("source_commit"),
                mac_algo=meta.get("mac_algo", ""),
                envelope_version=int(meta.get("envelope_version", 0)),
                authenticated=False, reason="envelope missing mac_hex"))
            continue
        # Recompute the MAC exactly the way protected_load does.
        meta_for_mac = {k: v for k, v in meta.items()
                          if k not in ("mac_hex", "mac_key_handle")}
        meta_bytes = json.dumps(meta_for_mac, sort_keys=True,
                                  separators=(",", ":")).encode("utf-8")
        try:
            mac_actual = _hmac_over(p, meta_bytes, keyref.key_bytes())
        except Exception as e:
            candidates.append(CheckpointCandidate(
                path=p, step=int(inner.get("step", -1)),
                authorized_step=int(meta.get("authorized_step", -1)),
                tokenizer_id=inner.get("tokenizer_id"),
                corpus_id=inner.get("corpus_id"),
                source_commit=meta.get("source_commit"),
                mac_algo=meta.get("mac_algo", ""),
                envelope_version=int(meta.get("envelope_version", 0)),
                authenticated=False,
                reason=f"MAC compute failed: {e}"))
            continue
        authenticated = hmac.compare_digest(mac_expected, mac_actual)
        candidates.append(CheckpointCandidate(
            path=p, step=int(inner.get("step", -1)),
            authorized_step=int(meta.get("authorized_step", -1)),
            tokenizer_id=inner.get("tokenizer_id"),
            corpus_id=inner.get("corpus_id"),
            source_commit=meta.get("source_commit"),
            mac_algo=meta.get("mac_algo", ""),
            envelope_version=int(meta.get("envelope_version", 0)),
            authenticated=authenticated,
            reason=None if authenticated else "MAC mismatch"))
    # Newest first (by step). Non-authenticated candidates sink to the end.
    candidates.sort(key=lambda c: (c.authenticated, c.step), reverse=True)
    return candidates


def latest_authenticated_checkpoint(
    checkpoint_dir: str, keyref
) -> Optional[CheckpointCandidate]:
    """Return the newest ``CheckpointCandidate`` whose ``authenticated``
    field is True, or None if there is no such checkpoint. Used by
    the launcher's Resume path to decide whether the Resume button is
    enabled at all."""
    for c in enumerate_checkpoints(checkpoint_dir, keyref):
        if c.authenticated:
            return c
    return None


@dataclass
class BuildableRecoveryDecision:
    """Convenience wrapper that assembles the fields the operator would
    otherwise have to type. The launcher's Recovery dialog fills in the
    reason field and the operator_authorization_ref; everything else is
    read from the selected candidate."""

    candidate: CheckpointCandidate
    reason: str
    operator_authorization_ref: str
    current_state_identity: str
    recovery_policy_version: int = 1

    def build(self):
        from aeon.protected_checkpoint import RecoveryDecision
        return RecoveryDecision(
            operator_authorization_ref=self.operator_authorization_ref,
            reason=self.reason,
            current_state_identity=self.current_state_identity,
            selected_state_identity=(
                f"sha256:{self.candidate.mac_algo}:step={self.candidate.step}"),
            integrity_result=("verified" if self.candidate.authenticated
                              else "unverified"),
            recovery_policy_version=self.recovery_policy_version,
            resulting_authorized_state=self.candidate.authorized_step,
        )

    def to_json(self) -> str:
        rd = self.build()
        return json.dumps(rd.__dict__, indent=2, sort_keys=True)
