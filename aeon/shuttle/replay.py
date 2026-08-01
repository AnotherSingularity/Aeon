"""ACIS-6 replay journal.

Records boundary events in order so that a Safe Stop → resume cycle
can prove the resumed run refuses to replay pre-stop boundaries.
The journal is transport-scoped: it records boundary index,
recursion epoch, broadcast id, semantic digest, and lifecycle
events — never raw tensor bytes.

Determinism guarantee: identical inputs, mode, and seeds produce
identical journals. This module does NOT provide replayability of
model weights; it provides a rejection surface so an operator can
detect a stale replay attempt across a checkpoint boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


class ReplayRefusal(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class ReplayRecord:
    boundary_index: int
    recursion_epoch: int
    broadcast_id: str
    semantic_digest: str
    kind: str  # PUBLISH | LEASE_ISSUE | LEASE_ACK | RETIRE


@dataclass
class ReplayJournal:
    _records: List[ReplayRecord] = field(default_factory=list)
    _last_epoch: int = -1
    _closed: bool = False
    _resume_boundary_floor: int = -1

    def __len__(self) -> int:
        return len(self._records)

    def append(self, rec: ReplayRecord) -> None:
        if self._closed:
            raise ReplayRefusal(
                "journal_closed",
                "cannot append after Safe Stop; resume required")
        if rec.recursion_epoch < self._last_epoch:
            raise ReplayRefusal(
                "epoch_regressed",
                f"{rec.recursion_epoch} < {self._last_epoch}")
        if rec.boundary_index <= self._resume_boundary_floor:
            raise ReplayRefusal(
                "boundary_replay_refused",
                f"boundary={rec.boundary_index} <= "
                f"floor={self._resume_boundary_floor}")
        self._records.append(rec)
        self._last_epoch = rec.recursion_epoch

    def close_for_safe_stop(self) -> None:
        self._closed = True

    def resume_from(self, boundary_floor: int) -> None:
        """Resume after checkpoint restore. Any boundary <= floor
        will be refused, so the operator cannot replay pre-stop
        broadcasts by accident."""
        if not self._closed:
            raise ReplayRefusal(
                "not_stopped",
                "must close_for_safe_stop before resuming")
        self._closed = False
        self._resume_boundary_floor = int(boundary_floor)

    def records(self) -> List[ReplayRecord]:
        return list(self._records)

    def last_boundary_index(self) -> int:
        return self._records[-1].boundary_index if self._records else -1
