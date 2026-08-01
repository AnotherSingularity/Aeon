"""ACIS-5 — lane + freshness + backpressure."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mk_capsule(cid):
    from aeon.shuttle.capsule import MutableStateCapsule, CapsuleState
    from aeon.shuttle.contracts import RepresentationContract
    return MutableStateCapsule(
        capsule_id=cid, lineage_id="l",
        source_zone="RECURSION", destination_zone="TRANSFORMER",
        current_mutable_owner="RECURSION",
        source_epoch=1, target_epoch=2,
        representation_contract=RepresentationContract(
            representation_id="r", source_zone="RECURSION",
            permitted_destinations=("TRANSFORMER",),
            shape=(1, 4), dtype="float32", device_class="cpu",
            model_identity="m", architecture_identity="a",
            semantic_basis_version=1,
            source_epoch=1, target_epoch=2, fixed_k=16,
            mutability="mutable",
            canonicalization_policy="contiguous_fp32_bytes"),
        causal_digest="sha256:x",
        lifecycle_state=CapsuleState.CREATED,
        expiration_epoch=100, maximum_hops=4,
        integrity_policy="hmac_authenticated",
        payload_handle=None,
    )


# ---------------------------------------------------------------------------
def test_lane_reserves_and_advances_in_order():
    from aeon.shuttle.lane import BucketLane, LaneStage
    lane = BucketLane(capacity=4)
    c = _mk_capsule("c1")
    slot = lane.reserve(c, admitted_epoch=0)
    assert slot.stage is LaneStage.RESERVED
    lane.advance("c1", LaneStage.STAGED)
    lane.advance("c1", LaneStage.VERIFIED)
    lane.advance("c1", LaneStage.DESTINATION)


def test_lane_refuses_skipping_stages():
    from aeon.shuttle.lane import BucketLane, LaneStage, LaneError
    lane = BucketLane(capacity=4)
    c = _mk_capsule("c2")
    lane.reserve(c, admitted_epoch=0)
    try:
        lane.advance("c2", LaneStage.VERIFIED)
    except LaneError as e:
        assert e.code == "invalid_lane_stage_transition"


def test_lane_capacity_enforced():
    from aeon.shuttle.lane import BucketLane, LaneError
    lane = BucketLane(capacity=1)
    lane.reserve(_mk_capsule("a"), admitted_epoch=0)
    try:
        lane.reserve(_mk_capsule("b"), admitted_epoch=1)
    except LaneError as e:
        assert e.code == "capacity_exceeded"


def test_lane_duplicate_suppression():
    from aeon.shuttle.lane import BucketLane, LaneError
    lane = BucketLane(capacity=4)
    lane.reserve(_mk_capsule("dup"), admitted_epoch=0)
    lane.advance("dup", __import__("aeon.shuttle.lane",
                                        fromlist=["LaneStage"]).LaneStage.STAGED)
    lane.advance("dup", __import__("aeon.shuttle.lane",
                                        fromlist=["LaneStage"]).LaneStage.VERIFIED)
    lane.advance("dup", __import__("aeon.shuttle.lane",
                                        fromlist=["LaneStage"]).LaneStage.DESTINATION)
    lane.pop_at_destination()
    try:
        lane.reserve(_mk_capsule("dup"), admitted_epoch=1)
    except LaneError as e:
        assert e.code == "duplicate_capsule"


def test_lane_fifo_pop_at_destination():
    from aeon.shuttle.lane import BucketLane, LaneStage
    lane = BucketLane(capacity=4)
    lane.reserve(_mk_capsule("first"), admitted_epoch=0)
    lane.reserve(_mk_capsule("second"), admitted_epoch=1)
    # Only when head reaches destination does pop return.
    assert lane.pop_at_destination() is None
    for s in (LaneStage.STAGED, LaneStage.VERIFIED, LaneStage.DESTINATION):
        lane.advance("first", s)
    popped = lane.pop_at_destination()
    assert popped.capsule_id == "first"


def test_lane_cancel_removes_capsule():
    from aeon.shuttle.lane import BucketLane
    lane = BucketLane(capacity=4)
    lane.reserve(_mk_capsule("x"), admitted_epoch=0)
    lane.cancel("x")
    assert len(lane) == 0


# ---------------------------------------------------------------------------
# Freshness
# ---------------------------------------------------------------------------
def test_freshness_rejects_stale():
    from aeon.shuttle.freshness import (
        FreshnessPolicy, enforce_freshness, FreshnessRejection,
    )
    policy = FreshnessPolicy(target_window_start=100,
                                target_window_end=200,
                                accept_causal_parent="sha256:root")
    try:
        enforce_freshness(source_recursion_epoch=50,
                            broadcast_id="b1",
                            causal_parent_digest="sha256:root",
                            expiration_epoch=1000,
                            current_epoch=101, policy=policy)
    except FreshnessRejection as e:
        assert e.code == "stale_source_epoch"


def test_freshness_rejects_future():
    from aeon.shuttle.freshness import (
        FreshnessPolicy, enforce_freshness, FreshnessRejection,
    )
    policy = FreshnessPolicy(target_window_start=100,
                                target_window_end=200,
                                accept_causal_parent="sha256:root")
    try:
        enforce_freshness(source_recursion_epoch=250,
                            broadcast_id="b1",
                            causal_parent_digest="sha256:root",
                            expiration_epoch=1000,
                            current_epoch=101, policy=policy)
    except FreshnessRejection as e:
        assert e.code == "future_source_epoch"


def test_freshness_rejects_duplicate():
    from aeon.shuttle.freshness import (
        FreshnessPolicy, enforce_freshness, FreshnessRejection,
    )
    policy = FreshnessPolicy(target_window_start=0, target_window_end=1000,
                                accept_causal_parent="sha256:root")
    enforce_freshness(source_recursion_epoch=100, broadcast_id="b1",
                        causal_parent_digest="sha256:root",
                        expiration_epoch=1000, current_epoch=100,
                        policy=policy)
    try:
        enforce_freshness(source_recursion_epoch=100, broadcast_id="b1",
                            causal_parent_digest="sha256:root",
                            expiration_epoch=1000, current_epoch=100,
                            policy=policy)
    except FreshnessRejection as e:
        assert e.code == "duplicate_admission"


def test_freshness_rejects_causal_mismatch():
    from aeon.shuttle.freshness import (
        FreshnessPolicy, enforce_freshness, FreshnessRejection,
    )
    policy = FreshnessPolicy(target_window_start=0, target_window_end=1000,
                                accept_causal_parent="sha256:parent-a")
    try:
        enforce_freshness(source_recursion_epoch=100, broadcast_id="b1",
                            causal_parent_digest="sha256:parent-B",
                            expiration_epoch=1000, current_epoch=100,
                            policy=policy)
    except FreshnessRejection as e:
        assert e.code == "causal_mismatch"


def test_freshness_rejects_expired():
    from aeon.shuttle.freshness import (
        FreshnessPolicy, enforce_freshness, FreshnessRejection,
    )
    policy = FreshnessPolicy(target_window_start=0, target_window_end=1000,
                                accept_causal_parent="sha256:root")
    try:
        enforce_freshness(source_recursion_epoch=100, broadcast_id="b1",
                            causal_parent_digest="sha256:root",
                            expiration_epoch=10, current_epoch=20,
                            policy=policy)
    except FreshnessRejection as e:
        assert e.code == "expired"


# ---------------------------------------------------------------------------
# Backpressure
# ---------------------------------------------------------------------------
def test_backpressure_bounds_admission_per_epoch():
    from aeon.shuttle.backpressure import BackpressureController
    ctl = BackpressureController(max_admit_per_epoch=2)
    ctl.advance_epoch(1)
    assert ctl.try_admit()
    assert ctl.try_admit()
    assert not ctl.try_admit()
    ctl.advance_epoch(2)
    assert ctl.try_admit()


def test_backpressure_refuses_epoch_regression():
    from aeon.shuttle.backpressure import (
        BackpressureController, BackpressureViolation,
    )
    ctl = BackpressureController()
    ctl.advance_epoch(2)
    try:
        ctl.advance_epoch(1)
    except BackpressureViolation as e:
        assert e.code == "epoch_regressed"


def test_backpressure_refuses_altering_K():
    from aeon.shuttle.backpressure import (
        BackpressureController, BackpressureViolation,
    )
    ctl = BackpressureController()
    ctl.assert_no_cognition_side_effect(k=16)
    try:
        ctl.assert_no_cognition_side_effect(k=8)
    except BackpressureViolation as e:
        assert e.code == "cognition_side_effect"


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
