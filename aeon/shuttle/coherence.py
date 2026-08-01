"""ACIS-6 coherence checker.

Verifies that a broadcast pair (Transformer + Substrate leases)
resolves to the SAME semantic identity and — under BUCKET mode
where both consumers hold a live tensor reference — the SAME
Python object. Divergence at either level is a coherence
violation and forces quarantine.

Coherence is a transport-level check; it does not alter model
state or gradients.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class CoherenceViolation(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def assert_pair_coherent(
    *,
    broadcast_id_t: str,
    broadcast_id_s: str,
    semantic_digest_t: str,
    semantic_digest_s: str,
    payload_t: Any,
    payload_s: Any,
) -> None:
    """Both leases must reference the same broadcast, the same
    semantic digest, and the same live tensor object."""
    if broadcast_id_t != broadcast_id_s:
        raise CoherenceViolation(
            "broadcast_id_divergence",
            f"t={broadcast_id_t!r} s={broadcast_id_s!r}")
    if semantic_digest_t != semantic_digest_s:
        raise CoherenceViolation(
            "semantic_digest_divergence",
            f"t={semantic_digest_t!r} s={semantic_digest_s!r}")
    if payload_t is not payload_s:
        raise CoherenceViolation(
            "payload_identity_divergence",
            "Transformer and Substrate must observe the SAME tensor "
            "object under BUCKET mode.")


@dataclass
class CoherenceLedger:
    """Optional running record of coherence assertions per boundary,
    used by the shuttle for post-hoc audit."""
    checked: int = 0
    violated: int = 0

    def observe_ok(self) -> None:
        self.checked += 1

    def observe_violation(self) -> None:
        self.checked += 1
        self.violated += 1
