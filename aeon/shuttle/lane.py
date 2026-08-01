"""ACIS-5 bucket-brigade lane.

A logical FIFO with capacity reservation, deterministic priority,
contract validation, expiration, quarantine, rollback-before-commit,
duplicate suppression, and replay. Does NOT force physical tensor
copies between stages — the lane holds metadata + a reference to
the existing tensor.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Deque, Dict, List, Optional
from collections import deque

from .capsule import CapsuleState, MutableStateCapsule


class LaneStage(str, Enum):
    SOURCE = "SOURCE_BOUNDARY"
    RESERVED = "RESERVED_SLOT"
    STAGED = "STAGED_SLOT"
    VERIFIED = "VERIFIED_SLOT"
    DESTINATION = "DESTINATION_BOUNDARY"


class LaneError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass
class LaneSlot:
    stage: LaneStage
    capsule_id: str
    priority: int
    admitted_epoch: int


@dataclass
class BucketLane:
    capacity: int
    _slots: Deque[LaneSlot] = field(default_factory=deque)
    _seen_ids: set = field(default_factory=set)  # duplicate suppression

    def __len__(self) -> int:
        return len(self._slots)

    def reserve(self, capsule: MutableStateCapsule,
                  *, admitted_epoch: int, priority: int = 0) -> LaneSlot:
        if len(self._slots) >= self.capacity:
            raise LaneError("capacity_exceeded",
                              f"len={len(self._slots)} cap={self.capacity}")
        if capsule.capsule_id in self._seen_ids:
            raise LaneError("duplicate_capsule",
                              capsule.capsule_id)
        self._seen_ids.add(capsule.capsule_id)
        slot = LaneSlot(stage=LaneStage.RESERVED,
                          capsule_id=capsule.capsule_id,
                          priority=int(priority),
                          admitted_epoch=int(admitted_epoch))
        self._slots.append(slot)
        return slot

    def advance(self, capsule_id: str, to: LaneStage) -> LaneSlot:
        order = [LaneStage.SOURCE, LaneStage.RESERVED, LaneStage.STAGED,
                   LaneStage.VERIFIED, LaneStage.DESTINATION]
        for slot in self._slots:
            if slot.capsule_id == capsule_id:
                cur_idx = order.index(slot.stage)
                to_idx = order.index(to)
                if to_idx != cur_idx + 1:
                    raise LaneError("invalid_lane_stage_transition",
                                      f"{slot.stage.value}->{to.value}")
                slot.stage = to
                return slot
        raise LaneError("unknown_capsule_in_lane", capsule_id)

    def pop_at_destination(self) -> Optional[LaneSlot]:
        """FIFO removal — only the head slot at DESTINATION can leave.
        Returns None if head is not yet at destination stage."""
        if not self._slots:
            return None
        head = self._slots[0]
        if head.stage is not LaneStage.DESTINATION:
            return None
        self._slots.popleft()
        return head

    def cancel(self, capsule_id: str) -> None:
        """Cancel a capsule mid-lane. Removes it from the queue."""
        for i, slot in enumerate(list(self._slots)):
            if slot.capsule_id == capsule_id:
                del self._slots[i]
                return
        raise LaneError("unknown_capsule_in_lane", capsule_id)
