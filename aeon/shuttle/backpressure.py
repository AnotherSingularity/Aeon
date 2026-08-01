"""ACIS-5 backpressure.

Backpressure controls TRANSPORT ONLY. It may not alter:
    * K (the slow-clock interval).
    * Model semantics.
    * Loss.
    * Token routing.
    * Substrate gate.
    * Recursion computation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


class BackpressureViolation(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass
class BackpressureController:
    """Bounds only transport-level fields (route, capacity, chunk_size,
    reservation policy, retry timing, prefetch distance, priority)."""

    max_admit_per_epoch: int = 64
    admitted_this_epoch: int = 0
    current_epoch: int = -1

    def advance_epoch(self, new_epoch: int) -> None:
        if new_epoch < self.current_epoch:
            raise BackpressureViolation("epoch_regressed",
                                          f"{new_epoch} < {self.current_epoch}")
        if new_epoch != self.current_epoch:
            self.current_epoch = new_epoch
            self.admitted_this_epoch = 0

    def try_admit(self) -> bool:
        if self.admitted_this_epoch >= self.max_admit_per_epoch:
            return False
        self.admitted_this_epoch += 1
        return True

    def assert_no_cognition_side_effect(self, k: int, expected_k: int = 16) -> None:
        """Prove that no attempt was made to change K under backpressure."""
        if k != expected_k:
            raise BackpressureViolation(
                "cognition_side_effect",
                f"backpressure attempted to change K to {k}; expected "
                f"{expected_k}")
