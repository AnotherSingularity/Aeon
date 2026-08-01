"""ACIS-4 mutable-ownership ledger.

Single-ownership invariant: for every mutable capsule q,
    sum over zones z of mutableOwner(q, z) == 1.

Immutable broadcasts are EXCLUDED from this count — they have zero
mutable owners and multiple read-only leases.

The ledger is authoritative after infrastructure failure (§6). It
records every ownership transfer with a chained digest so a replayed
transfer is detectable.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .capsule import (
    CapsuleError, CapsuleState, MutableStateCapsule, transition_capsule,
)


class OwnershipViolation(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class OwnershipEntry:
    seq: int
    capsule_id: str
    from_zone: str
    to_zone: str
    at_state: str
    prev_ledger: str
    ledger_digest: str
    ts_utc: str


@dataclass
class OwnershipLedger:
    _entries: List[OwnershipEntry] = field(default_factory=list)
    _current_owner: Dict[str, str] = field(default_factory=dict)
    _seq: int = 0

    def current_owner(self, capsule_id: str) -> Optional[str]:
        return self._current_owner.get(capsule_id)

    def head_digest(self) -> str:
        if not self._entries:
            return "sha256:" + "0" * 64
        return self._entries[-1].ledger_digest

    def record_create(self, capsule: MutableStateCapsule) -> OwnershipEntry:
        if capsule.capsule_id in self._current_owner:
            raise OwnershipViolation("capsule_already_created",
                                         capsule.capsule_id)
        self._current_owner[capsule.capsule_id] = capsule.current_mutable_owner
        return self._append(capsule.capsule_id, "-", capsule.current_mutable_owner,
                              capsule.lifecycle_state.value)

    def record_transfer(
        self,
        capsule: MutableStateCapsule,
        *,
        from_zone: str,
        to_zone: str,
        require_state: Optional[CapsuleState] = None,
    ) -> OwnershipEntry:
        """Transfer mutable authority. Refuses if:
            * capsule not previously created here.
            * the ledger's current owner disagrees with from_zone.
            * capsule.lifecycle_state doesn't match require_state.
            * to_zone would violate single-ownership (double-commit
              detection).
        """
        if capsule.capsule_id not in self._current_owner:
            raise OwnershipViolation(
                "unknown_capsule", capsule.capsule_id)
        if self._current_owner[capsule.capsule_id] != from_zone:
            raise OwnershipViolation(
                "wrong_source_owner",
                f"{capsule.capsule_id}: expected {from_zone!r} but "
                f"ledger says {self._current_owner[capsule.capsule_id]!r}")
        if require_state is not None and capsule.lifecycle_state is not require_state:
            raise OwnershipViolation(
                "wrong_lifecycle_state",
                f"{capsule.capsule_id}: state="
                f"{capsule.lifecycle_state.value} expected="
                f"{require_state.value}")
        if from_zone == to_zone:
            raise OwnershipViolation(
                "self_transfer_refused",
                f"{capsule.capsule_id}: {from_zone!r} -> {to_zone!r}")
        self._current_owner[capsule.capsule_id] = to_zone
        capsule.current_mutable_owner = to_zone
        return self._append(capsule.capsule_id, from_zone, to_zone,
                              capsule.lifecycle_state.value)

    def sum_owner_cardinality(self, capsule_id: str) -> int:
        """Return 1 if capsule_id has exactly one owner, 0 if none."""
        return 1 if capsule_id in self._current_owner else 0

    def _append(self, capsule_id: str, from_zone: str,
                  to_zone: str, at_state: str) -> OwnershipEntry:
        self._seq += 1
        prev = self.head_digest()
        material = "|".join([
            str(self._seq), capsule_id, from_zone, to_zone, at_state, prev,
        ]).encode("utf-8")
        digest = "sha256:" + hashlib.sha256(material).hexdigest()
        entry = OwnershipEntry(
            seq=self._seq, capsule_id=capsule_id,
            from_zone=from_zone, to_zone=to_zone,
            at_state=at_state, prev_ledger=prev,
            ledger_digest=digest,
            ts_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._entries.append(entry)
        return entry

    def entries(self) -> List[OwnershipEntry]:
        return list(self._entries)


def enforce_single_mutable_owner(
    capsules: List[MutableStateCapsule],
    ledger: OwnershipLedger,
) -> None:
    """Assert that every mutable capsule has exactly one owner. Immutable
    broadcasts are excluded from this check — they have zero mutable
    owners and are counted separately."""
    for c in capsules:
        if c.representation_contract.mutability != "mutable":
            continue
        n = ledger.sum_owner_cardinality(c.capsule_id)
        if n != 1:
            raise OwnershipViolation(
                "single_ownership_violated",
                f"{c.capsule_id}: owners={n}")
