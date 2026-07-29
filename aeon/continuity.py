"""
aeon/continuity.py — deterministic operating-state machine + analytical
manufacturing / communications abstractions + graceful degradation.

Directive F5. Aeon MAY recommend transitions; Aeon MAY NOT unilaterally
authorize consequential ones — every transition into CONTAINMENT / RECOVERY_* /
SAFE_HALT requires either an external operator authorization reference (§F5.1)
or a fail-closed condition from F4.6.

Torch-free. Manufacturing / comms are ANALYTICAL only — no control protocols,
no vendor commands, no offensive functionality.
"""
from __future__ import annotations

import enum
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


class State(enum.Enum):
    NORMAL = "NORMAL"
    ELEVATED_OBSERVATION = "ELEVATED_OBSERVATION"
    DEGRADED = "DEGRADED"
    CONTAINMENT = "CONTAINMENT"
    RECOVERY_PENDING = "RECOVERY_PENDING"
    RECOVERING = "RECOVERING"
    SAFE_HALT = "SAFE_HALT"


# ---------------------------------------------------------------------------
# Transition table (F5.1)
#
# key: (from_state, trigger_class) → allowed to_state.
# Aeon-side we validate the (from, trigger) → to; the OPERATOR-SIGNED
# authorization is required for consequential transitions.
# ---------------------------------------------------------------------------
CONSEQUENTIAL_TRANSITIONS = frozenset({
    State.CONTAINMENT, State.RECOVERING, State.SAFE_HALT
})

TRANSITION_TABLE: Dict[Tuple[State, str], State] = {
    # From NORMAL
    (State.NORMAL, "anomaly_observed"): State.ELEVATED_OBSERVATION,
    (State.NORMAL, "resource_pressure"): State.DEGRADED,
    (State.NORMAL, "integrity_failure"): State.CONTAINMENT,
    (State.NORMAL, "essential_guarantee_lost"): State.SAFE_HALT,

    # From ELEVATED_OBSERVATION
    (State.ELEVATED_OBSERVATION, "resolved"): State.NORMAL,
    (State.ELEVATED_OBSERVATION, "resource_pressure"): State.DEGRADED,
    (State.ELEVATED_OBSERVATION, "integrity_failure"): State.CONTAINMENT,
    (State.ELEVATED_OBSERVATION, "essential_guarantee_lost"): State.SAFE_HALT,

    # From DEGRADED
    (State.DEGRADED, "resolved"): State.NORMAL,
    (State.DEGRADED, "integrity_failure"): State.CONTAINMENT,
    (State.DEGRADED, "essential_guarantee_lost"): State.SAFE_HALT,

    # From CONTAINMENT
    (State.CONTAINMENT, "operator_recovery_authorized"): State.RECOVERY_PENDING,
    (State.CONTAINMENT, "essential_guarantee_lost"): State.SAFE_HALT,

    # From RECOVERY_PENDING
    (State.RECOVERY_PENDING, "recovery_started"): State.RECOVERING,
    (State.RECOVERY_PENDING, "essential_guarantee_lost"): State.SAFE_HALT,

    # From RECOVERING
    (State.RECOVERING, "recovery_verified"): State.NORMAL,
    (State.RECOVERING, "recovery_failed"): State.CONTAINMENT,
    (State.RECOVERING, "essential_guarantee_lost"): State.SAFE_HALT,

    # SAFE_HALT is terminal: only operator-driven restart leaves it.
    (State.SAFE_HALT, "operator_authorized_restart"): State.NORMAL,
}


PERMITTED_INITIATORS: Dict[str, str] = {
    "anomaly_observed": "aeon_analytical",
    "resource_pressure": "aeon_analytical",
    "integrity_failure": "aeon_or_operator",
    "essential_guarantee_lost": "aeon_or_operator",
    "resolved": "aeon_analytical",
    "operator_recovery_authorized": "operator",
    "recovery_started": "operator",
    "recovery_verified": "operator",
    "recovery_failed": "aeon_or_operator",
    "operator_authorized_restart": "operator",
}


ALLOWED_ACTIONS: Dict[State, List[str]] = {
    State.NORMAL: ["ordinary_training", "ordinary_inference", "ordinary_diagnostics",
                   "recommend_transition"],
    State.ELEVATED_OBSERVATION: ["ordinary_training", "ordinary_inference",
                                  "enhanced_sampling", "recommend_transition"],
    State.DEGRADED: ["reduced_batch", "reduced_seq_len", "reduced_diagnostics",
                     "recommend_transition"],
    State.CONTAINMENT: ["read_only_state", "audit_write", "await_operator"],
    State.RECOVERY_PENDING: ["read_only_state", "audit_write", "await_operator"],
    State.RECOVERING: ["strict_load", "audit_write", "verify_certificate"],
    State.SAFE_HALT: ["audit_write"],
}

PROHIBITED_ACTIONS: Dict[State, List[str]] = {
    State.CONTAINMENT: ["train", "checkpoint_save", "external_output"],
    State.RECOVERY_PENDING: ["train", "checkpoint_save", "external_output"],
    State.RECOVERING: ["train", "external_output"],
    State.SAFE_HALT: ["train", "inference", "checkpoint_save", "external_output"],
}


# ---------------------------------------------------------------------------
# Continuity controller
# ---------------------------------------------------------------------------
class ContinuityViolation(RuntimeError):
    """Raised on any unauthorised transition or on unauthorised action inside a state."""


@dataclass
class ContinuityController:
    state: State = State.NORMAL
    _history: List[Dict[str, Any]] = field(default_factory=list)

    def _emit_audit(self, kind: str, **payload: Any) -> None:
        self._history.append({"kind": kind, "state": self.state.value, **payload})

    def request_transition(
        self,
        trigger: str,
        *,
        evidence: Dict[str, Any],
        initiator: str,
        operator_authorization_ref: Optional[str] = None,
    ) -> State:
        """Attempt a state transition. Returns the new state on success; raises
        ContinuityViolation on any rule violation."""
        target = TRANSITION_TABLE.get((self.state, trigger))
        if target is None:
            raise ContinuityViolation(
                f"no transition from {self.state.value} on trigger {trigger!r}")

        permitted = PERMITTED_INITIATORS.get(trigger, "operator")
        if permitted == "operator" and initiator != "operator":
            raise ContinuityViolation(
                f"trigger {trigger!r} requires operator initiator; got {initiator!r}")
        if permitted == "aeon_or_operator" and initiator not in ("operator", "aeon_analytical"):
            raise ContinuityViolation(
                f"trigger {trigger!r} requires aeon_analytical or operator; got {initiator!r}")

        # Consequential transitions require an operator authorization reference,
        # UNLESS the transition is to SAFE_HALT because an essential guarantee
        # was lost (fail-closed § F4.6). Aeon MAY autonomously enter SAFE_HALT
        # to protect the environment; it MAY NOT autonomously exit it.
        if target in CONSEQUENTIAL_TRANSITIONS:
            self_authorized_safe_halt = (target is State.SAFE_HALT and
                                          trigger == "essential_guarantee_lost")
            self_authorized_containment = (target is State.CONTAINMENT and
                                            trigger == "integrity_failure")
            if not (self_authorized_safe_halt or self_authorized_containment):
                if not operator_authorization_ref:
                    raise ContinuityViolation(
                        f"transition to {target.value} requires operator_authorization_ref")

        old = self.state
        self.state = target
        self._emit_audit("transition", trigger=trigger, initiator=initiator,
                          from_state=old.value, to_state=target.value,
                          evidence_hash=_hash(evidence),
                          operator_authorization_ref=operator_authorization_ref)
        return target

    def check_action(self, action: str) -> None:
        """Refuse actions that are prohibited in the current state (§F5.1)."""
        prohibited = PROHIBITED_ACTIONS.get(self.state, [])
        if action in prohibited:
            raise ContinuityViolation(
                f"action {action!r} is PROHIBITED in state {self.state.value}")

    def history(self) -> List[Dict[str, Any]]:
        return list(self._history)


def _hash(obj: Any) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True,
                                      separators=(",", ":"), default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Manufacturing analytical abstractions (§F5.2) — RECOMMENDATIONS only.
# ---------------------------------------------------------------------------
@dataclass
class ManufacturingObservation:
    """Structured analytical observation from telemetry — never a control command."""
    kind: str            # e.g. "stale_telemetry", "sensor_disagreement", "process_drift"
    severity: str        # "info" | "elevated" | "degraded"
    evidence: Dict[str, Any]
    confidence: float    # 0..1
    missing_data: bool


def analyze_manufacturing_telemetry(frames: List[Dict[str, Any]],
                                     *,
                                     staleness_seconds: float = 30.0,
                                     process_drift_z: float = 3.0,
                                     ) -> List[ManufacturingObservation]:
    """Analytical only. Detects: stale telemetry, sensor disagreement, process
    drift (z-score against a rolling baseline supplied per frame), quality-
    control deviation, dependency/material interruption. Returns observations —
    never commands, never vendor-specific instructions."""
    obs: List[ManufacturingObservation] = []
    if not frames:
        return [ManufacturingObservation("no_data", "info", {}, 0.0, True)]

    now = max(f.get("ts", 0) for f in frames)
    for f in frames:
        age = float(now - float(f.get("ts", 0)))
        if age > staleness_seconds:
            obs.append(ManufacturingObservation(
                "stale_telemetry", "elevated",
                {"source_id": f.get("source_id"), "age_s": age}, 0.9, False))

    # sensor_disagreement: multiple sensors reporting different values for
    # the same signal beyond a permitted band
    by_signal: Dict[str, List[float]] = {}
    for f in frames:
        for sig, val in (f.get("signals") or {}).items():
            if isinstance(val, (int, float)):
                by_signal.setdefault(sig, []).append(float(val))
    for sig, vals in by_signal.items():
        if len(vals) >= 2:
            spread = max(vals) - min(vals)
            mean = sum(vals) / len(vals)
            if mean and abs(spread / (abs(mean) + 1e-9)) > 0.10:
                obs.append(ManufacturingObservation(
                    "sensor_disagreement", "elevated",
                    {"signal": sig, "spread": spread, "mean": mean}, 0.8, False))

    # process_drift: z-score against baseline supplied in the frame
    for f in frames:
        for sig, meta in (f.get("baselines") or {}).items():
            val = (f.get("signals") or {}).get(sig)
            mu = meta.get("mean"); sigma = meta.get("std")
            if isinstance(val, (int, float)) and isinstance(mu, (int, float)) and isinstance(sigma, (int, float)) and sigma > 0:
                z = abs((val - mu) / sigma)
                if z > process_drift_z:
                    obs.append(ManufacturingObservation(
                        "process_drift", "degraded" if z > 2 * process_drift_z else "elevated",
                        {"signal": sig, "z_score": z}, min(0.99, z / 10.0), False))

    # quality-control deviation
    for f in frames:
        qc = f.get("quality_control")
        if isinstance(qc, dict) and qc.get("out_of_spec"):
            obs.append(ManufacturingObservation(
                "quality_control_deviation", "degraded",
                {"details": qc}, 0.85, False))

    # dependency / material interruption
    for f in frames:
        for key in ("dependency_interrupted", "material_interrupted",
                    "schedule_degraded"):
            if f.get(key):
                obs.append(ManufacturingObservation(
                    key, "degraded", {"source_id": f.get("source_id")}, 0.8, False))
    return obs


# ---------------------------------------------------------------------------
# Communications analytical abstractions (§F5.3) — analytical only.
# ---------------------------------------------------------------------------
@dataclass
class CommunicationsObservation:
    kind: str
    severity: str
    evidence: Dict[str, Any]
    confidence: float
    missing_data: bool


def analyze_comms_telemetry(frames: List[Dict[str, Any]],
                             *,
                             freshness_s: float = 5.0,
                             latency_change_ratio: float = 3.0,
                             ) -> List[CommunicationsObservation]:
    obs: List[CommunicationsObservation] = []
    if not frames:
        return [CommunicationsObservation("no_data", "info", {}, 0.0, True)]

    # link availability
    for f in frames:
        if f.get("link_available") is False:
            obs.append(CommunicationsObservation(
                "link_unavailable", "degraded",
                {"link": f.get("link_id")}, 0.99, False))

    # message freshness
    now = max(f.get("ts", 0) for f in frames)
    for f in frames:
        age = float(now - float(f.get("ts", 0)))
        if age > freshness_s:
            obs.append(CommunicationsObservation(
                "stale_message", "elevated",
                {"link": f.get("link_id"), "age_s": age}, 0.85, False))

    # authentication status
    for f in frames:
        if f.get("authentication_status") == "failed":
            obs.append(CommunicationsObservation(
                "authentication_failure", "degraded",
                {"link": f.get("link_id")}, 0.99, False))

    # sequence continuity + duplicate/replay indicators
    for f in frames:
        if f.get("sequence_gap"):
            obs.append(CommunicationsObservation(
                "sequence_gap", "elevated",
                {"link": f.get("link_id")}, 0.9, False))
        if f.get("duplicate_indicator") or f.get("replay_indicator"):
            obs.append(CommunicationsObservation(
                "replay_or_duplicate", "elevated",
                {"link": f.get("link_id")}, 0.9, False))

    # latency change
    for f in frames:
        base = f.get("baseline_latency_ms"); cur = f.get("latency_ms")
        if isinstance(base, (int, float)) and isinstance(cur, (int, float)) and base > 0:
            if cur / base > latency_change_ratio:
                obs.append(CommunicationsObservation(
                    "latency_spike", "elevated",
                    {"link": f.get("link_id"), "cur_ms": cur, "base_ms": base}, 0.8, False))

    # bandwidth / queue / routing
    for f in frames:
        if f.get("bandwidth_pressure"):
            obs.append(CommunicationsObservation(
                "bandwidth_pressure", "elevated",
                {"link": f.get("link_id")}, 0.75, False))
        if f.get("queue_growth"):
            obs.append(CommunicationsObservation(
                "queue_growth", "elevated",
                {"link": f.get("link_id"), "queue_depth": f.get("queue_depth")}, 0.75, False))
        if f.get("route_degraded"):
            obs.append(CommunicationsObservation(
                "route_degraded", "elevated",
                {"link": f.get("link_id")}, 0.75, False))
    return obs


# ---------------------------------------------------------------------------
# Graceful degradation ordering (§F5.4)
# ---------------------------------------------------------------------------
DEGRADATION_ORDER = (
    "preserve_human_safety_boundaries",
    "preserve_certificate_and_integrity_checks",
    "preserve_artifact_authentication",
    "preserve_state_and_checkpoint_integrity",
    "preserve_critical_telemetry_validation",
    "preserve_essential_anomaly_detection",
    "reduce_optional_diagnostics",
    "reduce_nonessential_generation",
    "reduce_context_or_batch_within_validated_limits",
    "safe_halt_when_essentials_cannot_hold",
)


def degradation_plan(preserve_essentials: bool = True) -> List[str]:
    """Returns the (deterministic) ordered list of degradation steps. The
    directive REQUIRES security/integrity be preserved FIRST — this order
    encodes that requirement so downstream orchestrators cannot silently
    reorder it."""
    return list(DEGRADATION_ORDER)


# ---------------------------------------------------------------------------
# Simulation fixtures (§F5.5) — synthetic only
# ---------------------------------------------------------------------------
def synthetic_normal_manufacturing_frames(n: int = 5) -> List[Dict[str, Any]]:
    return [{"source_id": f"mfg_{i}", "ts": 100.0 + i,
             "signals": {"temp": 25.0 + 0.05 * i, "pressure": 1.0 + 0.001 * i},
             "baselines": {"temp": {"mean": 25.0, "std": 0.5},
                           "pressure": {"mean": 1.0, "std": 0.01}}}
            for i in range(n)]


def synthetic_stale_frame(source_id: str = "mfg_1", age_s: float = 60.0,
                            now: float = 200.0) -> Dict[str, Any]:
    return {"source_id": source_id, "ts": now - age_s,
            "signals": {"temp": 25.0}, "baselines": {}}


def synthetic_normal_comms_frames(n: int = 5) -> List[Dict[str, Any]]:
    return [{"link_id": "L0", "ts": 100.0 + i, "link_available": True,
             "authentication_status": "ok", "sequence_gap": False,
             "duplicate_indicator": False, "replay_indicator": False,
             "latency_ms": 40, "baseline_latency_ms": 40}
            for i in range(n)]
