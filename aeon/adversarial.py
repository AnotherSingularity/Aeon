"""
aeon/adversarial.py — F6 adversarial-resilience test harness.

Directive F6.6 requires each adversarial case to record:
  - Threat-model identifier
  - Preconditions
  - Injected fault or hostile condition
  - Expected response
  - Actual response
  - Detection result
  - Containment result
  - Recovery result
  - Audit-event identifier
  - Pass / fail

This module supplies a light dataclass + a run helper. The harness is DEFENSIVE
only: it invokes the SAME public interfaces an operator would (strict_load,
protected_load, corpus_manifest.refuse_if_invalid, runtime_policy.check_path)
and observes that they refuse.

Torch-free. Tests using this harness live in tests/test_adversarial_*.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class AdversarialCase:
    threat_id: str                                          # T01..T18
    category: str                                           # "artifact"|"data"|"runtime"|"model_state"|"availability"
    name: str
    precondition: str
    injection: str
    expected_response: str
    actual_response: Optional[str] = None
    detection: Optional[str] = None                          # "detected" | "missed"
    containment: Optional[str] = None                        # "contained" | "escaped" | "n/a"
    recovery: Optional[str] = None                           # "possible" | "not_required" | "impossible"
    audit_event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    passed: bool = False
    error: Optional[str] = None
    ts: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


def run_case(case: AdversarialCase, action: Callable[[], Any],
              *, expect_exception: type = Exception,
              expect_substr: Optional[str] = None) -> AdversarialCase:
    """Run an action expecting it to raise `expect_exception` (optionally with a
    substring in the message). Populates the case fields and returns it."""
    try:
        action()
        case.actual_response = "action_completed_no_error"
        case.detection = "missed"
        case.containment = "escaped"
        case.recovery = "not_required"
        case.passed = False
        return case
    except expect_exception as e:
        msg = str(e)
        case.actual_response = f"{type(e).__name__}: {msg[:200]}"
        if expect_substr and expect_substr not in msg:
            case.detection = "detected_wrong_reason"
            case.containment = "contained"
            case.recovery = "possible"
            case.passed = False
            case.error = f"expected substring {expect_substr!r} not in message"
            return case
        case.detection = "detected"
        case.containment = "contained"
        case.recovery = "possible"
        case.passed = True
        return case
    except BaseException as e:
        # Any BaseException that isn't the expected class
        case.actual_response = f"UNEXPECTED {type(e).__name__}: {str(e)[:200]}"
        case.detection = "detected_wrong_kind"
        case.containment = "unknown"
        case.recovery = "unknown"
        case.passed = False
        return case


def summarise(cases) -> Dict[str, Any]:
    return {
        "total": len(cases),
        "pass": sum(1 for c in cases if c.passed),
        "fail": sum(1 for c in cases if not c.passed),
        "by_category": {
            cat: sum(1 for c in cases if c.category == cat)
            for cat in {c.category for c in cases}
        },
    }
