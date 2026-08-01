"""ACIS-4 mutable-state capsule.

Distinct from ImmutableRecursionBroadcast. A capsule carries state
that DOES change hands (a mutable transfer between two zones); at
most one zone owns the mutation authority at any time.

The capsule's state machine is SEPARATE from the broadcast lease
state machine. Do not use one ambiguous state machine for both classes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .contracts import RepresentationContract


class CapsuleState(str, Enum):
    CREATED = "CREATED"
    RESERVED = "RESERVED"
    STAGED = "STAGED"
    VERIFIED = "VERIFIED"
    COMMITTED = "COMMITTED"
    AVAILABLE = "AVAILABLE"
    CONSUMED = "CONSUMED"

    # Failure states
    EXPIRED = "EXPIRED"
    REJECTED = "REJECTED"
    QUARANTINED = "QUARANTINED"
    ROLLED_BACK = "ROLLED_BACK"
    RECOMPUTE_REQUIRED = "RECOMPUTE_REQUIRED"


class CapsuleError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass
class MutableStateCapsule:
    capsule_id: str
    lineage_id: str
    source_zone: str
    destination_zone: str
    current_mutable_owner: str
    source_epoch: int
    target_epoch: int
    representation_contract: RepresentationContract
    causal_digest: str
    lifecycle_state: CapsuleState
    expiration_epoch: int
    maximum_hops: int
    integrity_policy: str
    payload_handle: Any  # LIVE reference; may change owner but not identity


def _valid_capsule_transitions() -> "dict[CapsuleState, set[CapsuleState]]":
    S = CapsuleState
    return {
        S.CREATED:  {S.RESERVED, S.EXPIRED, S.REJECTED, S.QUARANTINED},
        S.RESERVED: {S.STAGED, S.EXPIRED, S.REJECTED, S.QUARANTINED,
                      S.ROLLED_BACK},
        S.STAGED:   {S.VERIFIED, S.EXPIRED, S.REJECTED, S.QUARANTINED,
                      S.ROLLED_BACK, S.RECOMPUTE_REQUIRED},
        S.VERIFIED: {S.COMMITTED, S.QUARANTINED, S.ROLLED_BACK,
                      S.RECOMPUTE_REQUIRED},
        S.COMMITTED: {S.AVAILABLE, S.EXPIRED, S.QUARANTINED},
        S.AVAILABLE: {S.CONSUMED, S.EXPIRED, S.QUARANTINED},
        S.CONSUMED: set(),  # terminal
        S.EXPIRED: set(),
        S.REJECTED: set(),
        S.QUARANTINED: set(),
        S.ROLLED_BACK: {S.CREATED, S.RECOMPUTE_REQUIRED},
        S.RECOMPUTE_REQUIRED: {S.CREATED},
    }


def transition_capsule(capsule: MutableStateCapsule,
                          to: CapsuleState) -> MutableStateCapsule:
    allowed = _valid_capsule_transitions().get(
        capsule.lifecycle_state, set())
    if to not in allowed:
        raise CapsuleError(
            "invalid_capsule_transition",
            f"{capsule.lifecycle_state.value} -> {to.value}")
    capsule.lifecycle_state = to
    return capsule
