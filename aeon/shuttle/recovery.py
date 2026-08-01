"""ACIS-6 recovery controller.

Coordinates Safe Stop → protected-checkpoint save → resume with
the transport layer:

  1. initiate_safe_stop() marks the shuttle as draining. No new
     broadcasts may be published; in-flight lanes finish
     nominally or are cancelled.
  2. drain_lane() cancels all reserved slots on a given lane.
  3. close_for_checkpoint() snapshots the replay journal state
     (boundary floor) and closes the journal.
  4. resume_after_checkpoint() reopens the journal past the
     recorded floor, guaranteeing that a resumed run cannot
     replay pre-stop boundaries.

The recovery controller does NOT touch model state — that path
belongs to the protected checkpoint lifecycle in
aeon.recovery.protected_checkpoint (from W10 tranche). The
recovery controller talks to it via a small handoff API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .lane import BucketLane
from .replay import ReplayJournal


class RecoveryViolation(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass
class RecoveryController:
    replay_journal: ReplayJournal = field(default_factory=ReplayJournal)
    lanes: List[BucketLane] = field(default_factory=list)
    _draining: bool = False
    _stopped: bool = False
    _resume_floor: int = -1
    _checkpoint_callback: Optional[Callable[[], None]] = None

    def register_lane(self, lane: BucketLane) -> None:
        self.lanes.append(lane)

    def initiate_safe_stop(self) -> None:
        if self._stopped:
            raise RecoveryViolation("already_stopped")
        self._draining = True

    def is_draining(self) -> bool:
        return self._draining

    def drain_lane(self, lane: BucketLane) -> int:
        """Cancel every remaining slot on the lane. Returns the
        number cancelled."""
        if not self._draining:
            raise RecoveryViolation(
                "drain_without_stop",
                "initiate_safe_stop must be called first")
        cancelled = 0
        while len(lane) > 0:
            head = lane._slots[0]
            lane.cancel(head.capsule_id)
            cancelled += 1
        return cancelled

    def close_for_checkpoint(self) -> None:
        if not self._draining:
            raise RecoveryViolation("close_without_stop")
        for lane in self.lanes:
            if len(lane) > 0:
                raise RecoveryViolation(
                    "lane_not_drained",
                    f"{len(lane)} slots remain")
        self._resume_floor = self.replay_journal.last_boundary_index()
        self.replay_journal.close_for_safe_stop()
        self._stopped = True
        if self._checkpoint_callback is not None:
            self._checkpoint_callback()

    def resume_after_checkpoint(self) -> None:
        if not self._stopped:
            raise RecoveryViolation(
                "resume_without_stop",
                "close_for_checkpoint must be called first")
        self.replay_journal.resume_from(self._resume_floor)
        self._draining = False
        self._stopped = False

    def bind_checkpoint_callback(
            self, cb: Callable[[], None]) -> None:
        """Optional hook — invoked from close_for_checkpoint after
        the journal has been closed and the resume floor recorded.
        Intended for the protected-checkpoint save routine from
        the W10 tranche."""
        self._checkpoint_callback = cb
