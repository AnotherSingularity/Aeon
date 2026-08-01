"""ACIS rollout modes (§8).

Fail-closed: unknown mode strings raise.
"""
from __future__ import annotations

from enum import Enum


class ShuttleMode(str, Enum):
    OFF = "off"
    OBSERVE = "observe"
    BUCKET = "bucket"
    CONVEYOR_EXPERIMENTAL = "conveyor_experimental"


class UnknownShuttleMode(RuntimeError):
    """Raised on any string that is not a member of ShuttleMode."""


def parse_shuttle_mode(value: str) -> ShuttleMode:
    """Case-sensitive parse; unknown values fail closed."""
    if not isinstance(value, str):
        raise UnknownShuttleMode(f"expected str, got {type(value).__name__}")
    try:
        return ShuttleMode(value)
    except ValueError as e:
        raise UnknownShuttleMode(
            f"{value!r} not in {[m.value for m in ShuttleMode]}") from e


def is_default_off(mode: ShuttleMode) -> bool:
    return mode is ShuttleMode.OFF
