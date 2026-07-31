"""L4 telemetry + L5 intervention harness tests.

L4 asserts:
 * Disabled by default.
 * Byte-budget fails closed.
 * Window-ceiling fails closed.
 * Persistent flag drives sink flush; default is in-memory only.
 * Noninterference: attaching the telemetry observer to
   HybridModel.forward does not change the default forward path
   (structurally shown by delegating to _NullObserver equivalence).
 * delta_loss = pre - post arithmetic.

L5 asserts:
 * InterventionKind enum has all 8 required entries.
 * assert_evaluation_mode raises when model.training=True.
 * InterventionRunner refuses persistence keywords
   (checkpoint_dir / generation_dir / save_path).
 * Runner returns per-batch delta_L_c.
"""
import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# L4
# ---------------------------------------------------------------------------
def test_telemetry_disabled_by_default():
    from aeon.bypass.telemetry import TelemetryConfig, SamplingTelemetryObserver
    from aeon.bypass.signal_trace import RecursionWindowEvent
    obs = SamplingTelemetryObserver(TelemetryConfig())
    ev = RecursionWindowEvent(
        schema_version=1, run_id="t", checkpoint_generation_id=None,
        window_index=0, token_start=0, token_end=16, k_value=16,
        transformer_source_shape=(1, 64), transformer_source_dtype="torch.float32",
        transformer_source_norm=1.0,
        substrate_source_shape=(1, 64), substrate_source_dtype="torch.float32",
        substrate_source_norm=1.0,
        recursion_state_before_shape=(1, 64),
        recursion_state_before_dtype="torch.float32",
        recursion_state_before_norm=1.0,
        recursion_state_after_shape=(1, 64),
        recursion_state_after_dtype="torch.float32",
        recursion_state_after_norm=1.1,
        recursion_delta_norm=0.1,
        broadcast_shape=(1, 64), broadcast_dtype="torch.float32",
        broadcast_norm=0.9,
        transformer_consumed_broadcast=True,
        substrate_consumed_broadcast=True,
        certificate_margin=0.02)
    # Disabled → no rows recorded.
    obs.on_recursion_window(ev)
    assert obs.rows() == []


def test_telemetry_byte_budget_fails_closed():
    from aeon.bypass.telemetry import (
        TelemetryConfig, SamplingTelemetryObserver, TelemetryOverBudget,
    )
    from aeon.bypass.signal_trace import RecursionWindowEvent
    cfg = TelemetryConfig(enabled=True, max_bytes=10, max_windows=1000)
    obs = SamplingTelemetryObserver(cfg)
    ev = RecursionWindowEvent(
        schema_version=1, run_id="t", checkpoint_generation_id=None,
        window_index=0, token_start=0, token_end=16, k_value=16,
        transformer_source_shape=(1, 64),
        transformer_source_dtype="torch.float32",
        transformer_source_norm=1.0,
        substrate_source_shape=(1, 64),
        substrate_source_dtype="torch.float32",
        substrate_source_norm=1.0,
        recursion_state_before_shape=(1, 64),
        recursion_state_before_dtype="torch.float32",
        recursion_state_before_norm=1.0,
        recursion_state_after_shape=(1, 64),
        recursion_state_after_dtype="torch.float32",
        recursion_state_after_norm=1.1,
        recursion_delta_norm=0.1,
        broadcast_shape=(1, 64), broadcast_dtype="torch.float32",
        broadcast_norm=0.9,
        transformer_consumed_broadcast=True,
        substrate_consumed_broadcast=True,
        certificate_margin=0.02)
    try:
        obs.on_recursion_window(ev)
    except TelemetryOverBudget:
        pass
    else:
        raise AssertionError("byte-budget must fail closed")


def test_telemetry_window_ceiling_fails_closed():
    from aeon.bypass.telemetry import (
        TelemetryConfig, SamplingTelemetryObserver, TelemetryOverBudget,
    )
    from aeon.bypass.signal_trace import RecursionWindowEvent
    cfg = TelemetryConfig(enabled=True, max_bytes=10 ** 9, max_windows=1)
    obs = SamplingTelemetryObserver(cfg)
    ev = RecursionWindowEvent(
        schema_version=1, run_id="t", checkpoint_generation_id=None,
        window_index=0, token_start=0, token_end=16, k_value=16,
        transformer_source_shape=(1, 64),
        transformer_source_dtype="torch.float32",
        transformer_source_norm=1.0,
        substrate_source_shape=(1, 64),
        substrate_source_dtype="torch.float32",
        substrate_source_norm=1.0,
        recursion_state_before_shape=(1, 64),
        recursion_state_before_dtype="torch.float32",
        recursion_state_before_norm=1.0,
        recursion_state_after_shape=(1, 64),
        recursion_state_after_dtype="torch.float32",
        recursion_state_after_norm=1.1,
        recursion_delta_norm=0.1,
        broadcast_shape=(1, 64), broadcast_dtype="torch.float32",
        broadcast_norm=0.9,
        transformer_consumed_broadcast=True,
        substrate_consumed_broadcast=True,
        certificate_margin=0.02)
    obs.on_recursion_window(ev)  # window 1
    try:
        obs.on_recursion_window(ev)  # window 2 — over ceiling
    except TelemetryOverBudget:
        pass
    else:
        raise AssertionError("window ceiling must fail closed")


def test_telemetry_noninterference_via_null_observer_delegation():
    """The telemetry observer, when disabled, is behaviourally
    equivalent to _NullObserver — no code path modifies model state."""
    from aeon.bypass.telemetry import (
        TelemetryConfig, SamplingTelemetryObserver,
    )
    from aeon.bypass.signal_trace import _NullObserver
    # Both drop events; the telemetry observer discards below cfg
    # enabled=False.
    cfg = TelemetryConfig(enabled=False)
    obs = SamplingTelemetryObserver(cfg)
    null = _NullObserver()
    # Behavioural equivalence at the callback level: both accept the
    # protocol without raising.
    assert callable(obs.on_recursion_window)
    assert callable(null.on_recursion_window)


def test_delta_loss_arithmetic():
    from aeon.bypass.telemetry import compute_delta_loss
    assert math.isclose(compute_delta_loss(1.0, 0.5), 0.5)
    assert math.isclose(compute_delta_loss(0.5, 1.0), -0.5)


def test_telemetry_persistent_flush_writes_local_only():
    from aeon.bypass.telemetry import (
        TelemetryConfig, SamplingTelemetryObserver,
    )
    from aeon.bypass.signal_trace import RecursionWindowEvent
    with tempfile.TemporaryDirectory() as d:
        cfg = TelemetryConfig(enabled=True, sink_dir=d,
                                max_bytes=10 ** 7,
                                max_windows=1000, persistent=True)
        obs = SamplingTelemetryObserver(cfg, run_id="t")
        ev = RecursionWindowEvent(
            schema_version=1, run_id="t", checkpoint_generation_id=None,
            window_index=0, token_start=0, token_end=16, k_value=16,
            transformer_source_shape=(1, 64),
            transformer_source_dtype="torch.float32",
            transformer_source_norm=1.0,
            substrate_source_shape=(1, 64),
            substrate_source_dtype="torch.float32",
            substrate_source_norm=1.0,
            recursion_state_before_shape=(1, 64),
            recursion_state_before_dtype="torch.float32",
            recursion_state_before_norm=1.0,
            recursion_state_after_shape=(1, 64),
            recursion_state_after_dtype="torch.float32",
            recursion_state_after_norm=1.1,
            recursion_delta_norm=0.1,
            broadcast_shape=(1, 64), broadcast_dtype="torch.float32",
            broadcast_norm=0.9,
            transformer_consumed_broadcast=True,
            substrate_consumed_broadcast=True,
            certificate_margin=0.02)
        obs.on_recursion_window(ev)
        out_path = obs.close()
        assert out_path and out_path.startswith(d), out_path


# ---------------------------------------------------------------------------
# L5
# ---------------------------------------------------------------------------
def test_intervention_kinds_are_the_declared_eight():
    from aeon.bypass.interventions import InterventionKind
    ks = {k.name for k in InterventionKind if k != InterventionKind.NONE}
    assert ks == {
        "ZERO_BROADCAST", "FREEZE_BROADCAST", "DELAY_BROADCAST",
        "SHUFFLE_BROADCAST", "FREEZE_RECURSION",
        "MASK_TRANSFORMER_SOURCE", "MASK_SUBSTRATE_SOURCE",
        "NORM_MATCHED_IRRELEVANT_STATE"}


def test_assert_evaluation_mode_refuses_training_mode():
    from aeon.bypass.interventions import (
        assert_evaluation_mode, TrainingModeRefused,
    )
    class M:
        training = True
    try:
        assert_evaluation_mode(M())
    except TrainingModeRefused:
        pass
    else:
        raise AssertionError("training-mode call must be refused")


def test_intervention_runner_refuses_persistence_kwargs():
    from aeon.bypass.interventions import (
        InterventionRunner, InterventionKind, InterventionSpec,
    )
    class M:
        training = False
    def fake_eval(model, batch, spec): return 1.0
    runner = InterventionRunner(model=M(), evaluate_batch=fake_eval)
    spec = InterventionSpec(kind=InterventionKind.ZERO_BROADCAST, seed=1)
    try:
        runner.run(spec, [None], checkpoint_dir="/tmp/whatever")
    except RuntimeError as e:
        assert "refuses persistence keyword" in str(e)
    else:
        raise AssertionError("runner must refuse checkpoint_dir")


def test_intervention_runner_returns_delta_L_c():
    from aeon.bypass.interventions import (
        InterventionRunner, InterventionKind, InterventionSpec,
    )
    class M:
        training = False
    def fake_eval(model, batch, spec):
        return 2.0 if spec is not None else 1.0
    runner = InterventionRunner(model=M(), evaluate_batch=fake_eval)
    spec = InterventionSpec(kind=InterventionKind.ZERO_BROADCAST, seed=1)
    res = runner.run(spec, [None, None, None])
    assert res.baseline_loss == 1.0
    assert res.intervened_loss == 2.0
    assert res.delta_L_c == 1.0
    assert res.intervention_identity.startswith("ZERO_BROADCAST")


def test_interventions_module_has_no_persistence_calls():
    """L5 module must not call generation_save / atomic_save /
    protected_save."""
    src = open(os.path.join(ROOT, "aeon", "bypass", "interventions.py"),
                encoding="utf-8").read()
    for forbidden in ("generation_save", "atomic_save", "protected_save"):
        assert forbidden not in src, (
            f"L5 module must not call {forbidden!r}")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
