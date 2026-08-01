"""ACIS-6 quarantine registry.

Any capsule or broadcast that fails contract validation, integrity
verification, coherence, or freshness is quarantined here. A
quarantined artifact is NEVER re-admitted to the lane until it has
been explicitly cleared. The registry chains reason + evidence so
the auditor can reconstruct why an artifact was excluded.

Quarantine is a transport-level action. It does not roll back model
state; the source zone's mutation authority is restored via the
capsule ROLLED_BACK path in aeon.shuttle.capsule.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


class QuarantineViolation(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class QuarantineEntry:
    artifact_id: str
    reason_code: str
    reason_detail: str
    boundary_index: int
    recursion_epoch: int


@dataclass
class QuarantineRegistry:
    _entries: Dict[str, QuarantineEntry] = field(default_factory=dict)
    _history: List[QuarantineEntry] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self._entries)

    def quarantine(self, artifact_id: str, *, reason_code: str,
                   reason_detail: str = "", boundary_index: int = -1,
                   recursion_epoch: int = -1) -> QuarantineEntry:
        if artifact_id in self._entries:
            raise QuarantineViolation("already_quarantined", artifact_id)
        entry = QuarantineEntry(
            artifact_id=artifact_id,
            reason_code=reason_code,
            reason_detail=reason_detail,
            boundary_index=int(boundary_index),
            recursion_epoch=int(recursion_epoch))
        self._entries[artifact_id] = entry
        self._history.append(entry)
        return entry

    def is_quarantined(self, artifact_id: str) -> bool:
        return artifact_id in self._entries

    def get(self, artifact_id: str) -> Optional[QuarantineEntry]:
        return self._entries.get(artifact_id)

    def clear(self, artifact_id: str) -> None:
        """Explicit operator action — remove a quarantined artifact.
        History remains in _history for audit."""
        if artifact_id not in self._entries:
            raise QuarantineViolation("not_quarantined", artifact_id)
        del self._entries[artifact_id]

    def refuse_readmission(self, artifact_id: str) -> None:
        """Called by the lane / freshness path before reserving a slot."""
        if artifact_id in self._entries:
            entry = self._entries[artifact_id]
            raise QuarantineViolation(
                "readmission_refused",
                f"{artifact_id} quarantined for {entry.reason_code}")

    def history(self) -> List[QuarantineEntry]:
        return list(self._history)
