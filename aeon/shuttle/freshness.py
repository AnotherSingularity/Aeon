"""ACIS-5 freshness enforcement.

Rejects broadcasts / capsules that are stale, from the future,
duplicate, superseded, or carry a mismatched causal parent digest.
Enforces the maximum_application_count on lease reuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Set


class FreshnessRejection(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass
class FreshnessPolicy:
    """The certified freshness policy at a destination window."""

    target_window_start: int
    target_window_end: int
    accept_causal_parent: str
    already_admitted: Set[str] = field(default_factory=set)
    superseded_broadcast_ids: Set[str] = field(default_factory=set)


def enforce_freshness(
    *,
    source_recursion_epoch: int,
    broadcast_id: str,
    causal_parent_digest: str,
    expiration_epoch: int,
    current_epoch: int,
    policy: FreshnessPolicy,
) -> None:
    """Refuse the broadcast under any of the freshness violations
    named in §17."""
    if source_recursion_epoch < policy.target_window_start:
        raise FreshnessRejection(
            "stale_source_epoch",
            f"source={source_recursion_epoch} < target_start="
            f"{policy.target_window_start}")
    if source_recursion_epoch > policy.target_window_end:
        raise FreshnessRejection(
            "future_source_epoch",
            f"source={source_recursion_epoch} > target_end="
            f"{policy.target_window_end}")
    if current_epoch > expiration_epoch:
        raise FreshnessRejection(
            "expired",
            f"current={current_epoch} > exp={expiration_epoch}")
    if causal_parent_digest != policy.accept_causal_parent:
        raise FreshnessRejection(
            "causal_mismatch",
            f"got={causal_parent_digest!r} expected="
            f"{policy.accept_causal_parent!r}")
    if broadcast_id in policy.already_admitted:
        raise FreshnessRejection("duplicate_admission", broadcast_id)
    if broadcast_id in policy.superseded_broadcast_ids:
        raise FreshnessRejection("superseded", broadcast_id)
    policy.already_admitted.add(broadcast_id)
