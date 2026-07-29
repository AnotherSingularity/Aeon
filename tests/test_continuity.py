"""
F5 — Defensive continuity framework tests.

Covers deterministic state transitions, prohibited-action enforcement,
manufacturing/comms analytical behaviours, and the graceful-degradation order.
Torch-free.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- §F5.1 state machine ----------------------------------------------------
def test_all_seven_states_present():
    from aeon.continuity import State
    names = {s.value for s in State}
    assert names == {"NORMAL", "ELEVATED_OBSERVATION", "DEGRADED", "CONTAINMENT",
                     "RECOVERY_PENDING", "RECOVERING", "SAFE_HALT"}


def test_deterministic_transition_from_normal():
    from aeon.continuity import ContinuityController, State
    c = ContinuityController()
    new = c.request_transition("anomaly_observed", evidence={"a": 1},
                                initiator="aeon_analytical")
    assert new == State.ELEVATED_OBSERVATION
    # resolved returns to NORMAL
    new = c.request_transition("resolved", evidence={"a": 2},
                                initiator="aeon_analytical")
    assert new == State.NORMAL


def test_aeon_cannot_unilaterally_authorize_recovery():
    from aeon.continuity import ContinuityController, ContinuityViolation, State
    c = ContinuityController()
    # get into CONTAINMENT via integrity_failure (aeon-authorised fail-closed)
    c.request_transition("integrity_failure", evidence={}, initiator="aeon_analytical")
    assert c.state == State.CONTAINMENT
    # RECOVERY requires operator authorization
    try:
        c.request_transition("operator_recovery_authorized",
                              evidence={"ref": "OP-1"}, initiator="aeon_analytical")
        assert False, "aeon-initiated recovery should be refused"
    except ContinuityViolation:
        pass
    # But an operator can
    c.request_transition("operator_recovery_authorized",
                          evidence={"ref": "OP-1"}, initiator="operator",
                          operator_authorization_ref="OP-1")
    assert c.state == State.RECOVERY_PENDING


def test_aeon_can_enter_safe_halt_on_essential_guarantee_lost():
    from aeon.continuity import ContinuityController, State
    c = ContinuityController()
    # Aeon MAY enter SAFE_HALT autonomously to protect the environment
    c.request_transition("essential_guarantee_lost", evidence={}, initiator="aeon_analytical")
    assert c.state == State.SAFE_HALT


def test_safe_halt_exit_requires_operator():
    from aeon.continuity import ContinuityController, ContinuityViolation, State
    c = ContinuityController()
    c.request_transition("essential_guarantee_lost", evidence={}, initiator="aeon_analytical")
    try:
        c.request_transition("operator_authorized_restart", evidence={},
                              initiator="aeon_analytical")
        assert False
    except ContinuityViolation:
        pass
    c.request_transition("operator_authorized_restart", evidence={},
                          initiator="operator", operator_authorization_ref="OP-2")
    assert c.state == State.NORMAL


def test_prohibited_action_refused_in_containment():
    from aeon.continuity import ContinuityController, ContinuityViolation
    c = ContinuityController()
    c.request_transition("integrity_failure", evidence={}, initiator="aeon_analytical")
    try:
        c.check_action("train")
        assert False, "train should be prohibited in CONTAINMENT"
    except ContinuityViolation:
        pass
    # audit_write is allowed
    c.check_action("audit_write")


def test_transition_history_recorded():
    from aeon.continuity import ContinuityController
    c = ContinuityController()
    c.request_transition("anomaly_observed", evidence={}, initiator="aeon_analytical")
    c.request_transition("resolved", evidence={}, initiator="aeon_analytical")
    hist = c.history()
    assert len(hist) >= 2
    assert hist[-1]["to_state"] == "NORMAL"


# ---- §F5.2 manufacturing analytics ------------------------------------------
def test_manufacturing_no_data_returns_no_data_observation():
    from aeon.continuity import analyze_manufacturing_telemetry
    obs = analyze_manufacturing_telemetry([])
    assert len(obs) == 1 and obs[0].kind == "no_data" and obs[0].missing_data


def test_manufacturing_detects_staleness():
    from aeon.continuity import analyze_manufacturing_telemetry, synthetic_stale_frame
    frames = [synthetic_stale_frame(age_s=60, now=200)]
    frames.append({"source_id": "fresh", "ts": 200, "signals": {"temp": 25}})
    obs = analyze_manufacturing_telemetry(frames, staleness_seconds=30)
    assert any(o.kind == "stale_telemetry" for o in obs)


def test_manufacturing_detects_process_drift():
    from aeon.continuity import analyze_manufacturing_telemetry
    frames = [{"source_id": "m1", "ts": 100, "signals": {"temp": 30.0},
               "baselines": {"temp": {"mean": 25.0, "std": 0.5}}}]
    obs = analyze_manufacturing_telemetry(frames, process_drift_z=3.0)
    assert any(o.kind == "process_drift" and o.evidence["z_score"] > 3.0 for o in obs)


def test_manufacturing_normal_frames_produce_no_alerts():
    from aeon.continuity import analyze_manufacturing_telemetry, synthetic_normal_manufacturing_frames
    obs = analyze_manufacturing_telemetry(synthetic_normal_manufacturing_frames(5))
    assert not any(o.severity == "degraded" for o in obs)


# ---- §F5.3 comms analytics --------------------------------------------------
def test_comms_detects_link_unavailability():
    from aeon.continuity import analyze_comms_telemetry
    obs = analyze_comms_telemetry([{"link_id": "L0", "ts": 100, "link_available": False}])
    assert any(o.kind == "link_unavailable" for o in obs)


def test_comms_detects_authentication_failure_and_replay():
    from aeon.continuity import analyze_comms_telemetry
    frames = [{"link_id": "L1", "ts": 100, "link_available": True,
                "authentication_status": "failed", "replay_indicator": True}]
    obs = analyze_comms_telemetry(frames)
    kinds = {o.kind for o in obs}
    assert "authentication_failure" in kinds
    assert "replay_or_duplicate" in kinds


def test_comms_normal_frames_produce_no_degraded_alerts():
    from aeon.continuity import analyze_comms_telemetry, synthetic_normal_comms_frames
    obs = analyze_comms_telemetry(synthetic_normal_comms_frames(5))
    assert not any(o.severity == "degraded" for o in obs)


# ---- §F5.4 graceful degradation --------------------------------------------
def test_degradation_order_preserves_security_first():
    from aeon.continuity import DEGRADATION_ORDER
    order = list(DEGRADATION_ORDER)
    # Security & integrity items MUST precede any "reduce" or "reduce_" item.
    security_indices = [i for i, x in enumerate(order) if "preserve_" in x]
    reduce_indices = [i for i, x in enumerate(order) if x.startswith("reduce_")]
    assert max(security_indices) < min(reduce_indices), order
    # SAFE_HALT is last
    assert order[-1] == "safe_halt_when_essentials_cannot_hold"


# ---- §F5.5 simulation harness -----------------------------------------------
def test_simulation_normal_manufacturing_valid():
    from aeon.continuity import (synthetic_normal_manufacturing_frames,
                                  analyze_manufacturing_telemetry)
    frames = synthetic_normal_manufacturing_frames(3)
    obs = analyze_manufacturing_telemetry(frames)
    assert not any(o.severity == "degraded" for o in obs)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
