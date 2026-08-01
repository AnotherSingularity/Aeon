"""ACIS-6 — recovery, replay, quarantine, coherence."""
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
# Quarantine
# ---------------------------------------------------------------------------
def test_quarantine_records_and_blocks_readmission():
    from aeon.shuttle.quarantine import (
        QuarantineRegistry, QuarantineViolation,
    )
    q = QuarantineRegistry()
    q.quarantine("cap-1", reason_code="contract_violation",
                     reason_detail="fixed_k!=16",
                     boundary_index=5, recursion_epoch=3)
    assert q.is_quarantined("cap-1")
    try:
        q.refuse_readmission("cap-1")
    except QuarantineViolation as e:
        assert e.code == "readmission_refused"


def test_quarantine_refuses_double_registration():
    from aeon.shuttle.quarantine import (
        QuarantineRegistry, QuarantineViolation,
    )
    q = QuarantineRegistry()
    q.quarantine("a", reason_code="bad")
    try:
        q.quarantine("a", reason_code="bad-again")
    except QuarantineViolation as e:
        assert e.code == "already_quarantined"


def test_quarantine_clear_removes_but_keeps_history():
    from aeon.shuttle.quarantine import QuarantineRegistry
    q = QuarantineRegistry()
    q.quarantine("z", reason_code="test")
    q.clear("z")
    assert not q.is_quarantined("z")
    assert len(q.history()) == 1


# ---------------------------------------------------------------------------
# Coherence
# ---------------------------------------------------------------------------
def test_coherence_accepts_identical_pair():
    from aeon.shuttle.coherence import assert_pair_coherent
    payload = object()
    assert_pair_coherent(
        broadcast_id_t="bx", broadcast_id_s="bx",
        semantic_digest_t="sha256:d", semantic_digest_s="sha256:d",
        payload_t=payload, payload_s=payload,
    )


def test_coherence_refuses_broadcast_divergence():
    from aeon.shuttle.coherence import (
        assert_pair_coherent, CoherenceViolation,
    )
    payload = object()
    try:
        assert_pair_coherent(
            broadcast_id_t="bx", broadcast_id_s="by",
            semantic_digest_t="d", semantic_digest_s="d",
            payload_t=payload, payload_s=payload,
        )
    except CoherenceViolation as e:
        assert e.code == "broadcast_id_divergence"


def test_coherence_refuses_digest_divergence():
    from aeon.shuttle.coherence import (
        assert_pair_coherent, CoherenceViolation,
    )
    payload = object()
    try:
        assert_pair_coherent(
            broadcast_id_t="bx", broadcast_id_s="bx",
            semantic_digest_t="d1", semantic_digest_s="d2",
            payload_t=payload, payload_s=payload,
        )
    except CoherenceViolation as e:
        assert e.code == "semantic_digest_divergence"


def test_coherence_refuses_object_divergence():
    from aeon.shuttle.coherence import (
        assert_pair_coherent, CoherenceViolation,
    )
    try:
        assert_pair_coherent(
            broadcast_id_t="bx", broadcast_id_s="bx",
            semantic_digest_t="d", semantic_digest_s="d",
            payload_t=object(), payload_s=object(),
        )
    except CoherenceViolation as e:
        assert e.code == "payload_identity_divergence"


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------
def test_replay_journal_appends_in_order():
    from aeon.shuttle.replay import ReplayJournal, ReplayRecord
    j = ReplayJournal()
    j.append(ReplayRecord(boundary_index=0, recursion_epoch=1,
                              broadcast_id="b0",
                              semantic_digest="d0", kind="PUBLISH"))
    j.append(ReplayRecord(boundary_index=1, recursion_epoch=1,
                              broadcast_id="b1",
                              semantic_digest="d1", kind="PUBLISH"))
    assert len(j) == 2


def test_replay_refuses_epoch_regression():
    from aeon.shuttle.replay import (
        ReplayJournal, ReplayRecord, ReplayRefusal,
    )
    j = ReplayJournal()
    j.append(ReplayRecord(boundary_index=0, recursion_epoch=3,
                              broadcast_id="b", semantic_digest="d",
                              kind="PUBLISH"))
    try:
        j.append(ReplayRecord(boundary_index=1, recursion_epoch=2,
                                  broadcast_id="b2", semantic_digest="d",
                                  kind="PUBLISH"))
    except ReplayRefusal as e:
        assert e.code == "epoch_regressed"


def test_replay_refuses_boundary_replay_after_resume():
    from aeon.shuttle.replay import (
        ReplayJournal, ReplayRecord, ReplayRefusal,
    )
    j = ReplayJournal()
    j.append(ReplayRecord(boundary_index=0, recursion_epoch=1,
                              broadcast_id="b0", semantic_digest="d",
                              kind="PUBLISH"))
    j.append(ReplayRecord(boundary_index=1, recursion_epoch=1,
                              broadcast_id="b1", semantic_digest="d",
                              kind="PUBLISH"))
    j.close_for_safe_stop()
    j.resume_from(1)
    try:
        j.append(ReplayRecord(boundary_index=1, recursion_epoch=1,
                                  broadcast_id="b1", semantic_digest="d",
                                  kind="PUBLISH"))
    except ReplayRefusal as e:
        assert e.code == "boundary_replay_refused"


def test_replay_refuses_append_while_closed():
    from aeon.shuttle.replay import (
        ReplayJournal, ReplayRecord, ReplayRefusal,
    )
    j = ReplayJournal()
    j.close_for_safe_stop()
    try:
        j.append(ReplayRecord(boundary_index=0, recursion_epoch=0,
                                  broadcast_id="b", semantic_digest="d",
                                  kind="PUBLISH"))
    except ReplayRefusal as e:
        assert e.code == "journal_closed"


# ---------------------------------------------------------------------------
# Recovery controller
# ---------------------------------------------------------------------------
def test_recovery_drain_cancels_lane_slots():
    from aeon.shuttle.lane import BucketLane
    from aeon.shuttle.recovery import RecoveryController
    lane = BucketLane(capacity=4)
    lane.reserve(_mk_capsule("x"), admitted_epoch=0)
    lane.reserve(_mk_capsule("y"), admitted_epoch=0)
    ctl = RecoveryController()
    ctl.register_lane(lane)
    ctl.initiate_safe_stop()
    n = ctl.drain_lane(lane)
    assert n == 2
    assert len(lane) == 0


def test_recovery_refuses_drain_without_stop():
    from aeon.shuttle.lane import BucketLane
    from aeon.shuttle.recovery import (
        RecoveryController, RecoveryViolation,
    )
    lane = BucketLane(capacity=4)
    ctl = RecoveryController()
    ctl.register_lane(lane)
    try:
        ctl.drain_lane(lane)
    except RecoveryViolation as e:
        assert e.code == "drain_without_stop"


def test_recovery_close_refuses_undrained_lane():
    from aeon.shuttle.lane import BucketLane
    from aeon.shuttle.recovery import (
        RecoveryController, RecoveryViolation,
    )
    lane = BucketLane(capacity=4)
    lane.reserve(_mk_capsule("still-here"), admitted_epoch=0)
    ctl = RecoveryController()
    ctl.register_lane(lane)
    ctl.initiate_safe_stop()
    try:
        ctl.close_for_checkpoint()
    except RecoveryViolation as e:
        assert e.code == "lane_not_drained"


def test_recovery_resume_rejects_prestop_boundary():
    from aeon.shuttle.lane import BucketLane
    from aeon.shuttle.recovery import RecoveryController
    from aeon.shuttle.replay import ReplayRecord, ReplayRefusal
    lane = BucketLane(capacity=4)
    ctl = RecoveryController()
    ctl.register_lane(lane)
    ctl.replay_journal.append(ReplayRecord(
        boundary_index=7, recursion_epoch=2,
        broadcast_id="b7", semantic_digest="d", kind="PUBLISH"))
    ctl.initiate_safe_stop()
    ctl.close_for_checkpoint()
    ctl.resume_after_checkpoint()
    try:
        ctl.replay_journal.append(ReplayRecord(
            boundary_index=7, recursion_epoch=2,
            broadcast_id="b7", semantic_digest="d", kind="PUBLISH"))
    except ReplayRefusal as e:
        assert e.code == "boundary_replay_refused"


def test_recovery_checkpoint_callback_invoked_once():
    from aeon.shuttle.recovery import RecoveryController
    calls = []
    ctl = RecoveryController()
    ctl.bind_checkpoint_callback(lambda: calls.append("saved"))
    ctl.initiate_safe_stop()
    ctl.close_for_checkpoint()
    assert calls == ["saved"]


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
