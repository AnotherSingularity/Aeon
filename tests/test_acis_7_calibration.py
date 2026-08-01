"""ACIS-7 — calibration + conveyor decision gates."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _sample(i, fwd, tr, ttb=0.0):
    from aeon.shuttle.calibration import TransportSample
    return TransportSample(boundary_index=i, forward_ms=fwd,
                              transport_ms=tr,
                              time_to_first_broadcast_ms=ttb)


def test_summarize_computes_overhead_fraction():
    from aeon.shuttle.calibration import summarize
    samples = [_sample(0, 100.0, 0.5), _sample(1, 100.0, 0.5)]
    r = summarize(samples)
    assert abs(r.transport_overhead_fraction - 0.005) < 1e-9
    assert r.is_bucket_certifiable()


def test_summarize_refuses_no_samples():
    from aeon.shuttle.calibration import (
        summarize, CalibrationRefusal,
    )
    try:
        summarize([])
    except CalibrationRefusal as e:
        assert e.code == "no_samples"


def test_bucket_over_budget_is_not_certifiable():
    from aeon.shuttle.calibration import summarize
    samples = [_sample(0, 100.0, 2.0)]  # 2% overhead
    r = summarize(samples)
    assert not r.is_bucket_certifiable(max_overhead_fraction=0.01)


def test_conveyor_refused_when_bucket_over_budget():
    from aeon.shuttle.calibration import summarize, decide_conveyor
    bucket = summarize([_sample(0, 100.0, 2.0)])   # 2% overhead
    d = decide_conveyor(bucket_report=bucket)
    assert d.decision == "conveyor_refused"
    assert d.reason_code == "bucket_overhead_too_high"


def test_conveyor_refused_when_no_conveyor_evidence():
    from aeon.shuttle.calibration import summarize, decide_conveyor
    bucket = summarize([_sample(0, 100.0, 0.5)])
    d = decide_conveyor(bucket_report=bucket)
    assert d.decision == "conveyor_refused"
    assert d.reason_code == "no_conveyor_evidence"


def test_conveyor_refused_when_slower():
    from aeon.shuttle.calibration import summarize, decide_conveyor
    bucket = summarize([_sample(0, 100.0, 0.5)])
    conv = summarize([_sample(0, 100.0, 0.9)])
    d = decide_conveyor(bucket_report=bucket, conveyor_report=conv,
                            conveyor_semantic_identity_preserved=True,
                            conveyor_autograd_identity_preserved=True)
    assert d.decision == "conveyor_refused"
    assert d.reason_code == "conveyor_slower_than_bucket"


def test_conveyor_refused_when_semantic_divergence():
    from aeon.shuttle.calibration import summarize, decide_conveyor
    bucket = summarize([_sample(0, 100.0, 0.5)])
    conv = summarize([_sample(0, 100.0, 0.4)])
    d = decide_conveyor(bucket_report=bucket, conveyor_report=conv,
                            conveyor_semantic_identity_preserved=False,
                            conveyor_autograd_identity_preserved=True)
    assert d.decision == "conveyor_refused"
    assert d.reason_code == "conveyor_semantic_divergence"


def test_conveyor_refused_when_autograd_divergence():
    from aeon.shuttle.calibration import summarize, decide_conveyor
    bucket = summarize([_sample(0, 100.0, 0.5)])
    conv = summarize([_sample(0, 100.0, 0.4)])
    d = decide_conveyor(bucket_report=bucket, conveyor_report=conv,
                            conveyor_semantic_identity_preserved=True,
                            conveyor_autograd_identity_preserved=False)
    assert d.decision == "conveyor_refused"
    assert d.reason_code == "conveyor_autograd_divergence"


def test_conveyor_certified_only_when_all_gates_pass():
    from aeon.shuttle.calibration import summarize, decide_conveyor
    bucket = summarize([_sample(0, 100.0, 0.8)])   # 0.8% overhead
    conv = summarize([_sample(0, 100.0, 0.5)])     # faster
    d = decide_conveyor(bucket_report=bucket, conveyor_report=conv,
                            conveyor_semantic_identity_preserved=True,
                            conveyor_autograd_identity_preserved=True)
    assert d.decision == "conveyor_certified"
    assert d.reason_code == "all_gates_passed"


def test_conveyor_default_state_in_acis_status_is_refused():
    """The V0.02 certified default is BUCKET, conveyor REFUSED
    until measured. This test locks that in acis_status.json."""
    import json, os as _os
    root = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    p = _os.path.join(root, "docs", "acis", "acis_status.json")
    with open(p, encoding="utf-8") as f:
        st = json.load(f)
    assert st["conveyor_certified"] is False


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
