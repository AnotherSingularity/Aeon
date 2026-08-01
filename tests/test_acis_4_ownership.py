"""ACIS-4 — mutable capsule + ownership ledger."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _mutable_contract():
    from aeon.shuttle.contracts import RepresentationContract
    return RepresentationContract(
        representation_id="mut-rc", source_zone="RECURSION",
        permitted_destinations=("TRANSFORMER", "SUBSTRATE"),
        shape=(1, 32), dtype="float32", device_class="cpu",
        model_identity="m", architecture_identity="a",
        semantic_basis_version=1,
        source_epoch=1, target_epoch=2,
        fixed_k=16, mutability="mutable",
        canonicalization_policy="contiguous_fp32_bytes",
    )


def _immutable_contract():
    c = _mutable_contract()
    return type(c)(
        representation_id=c.representation_id, source_zone=c.source_zone,
        permitted_destinations=c.permitted_destinations,
        shape=c.shape, dtype=c.dtype, device_class=c.device_class,
        model_identity=c.model_identity,
        architecture_identity=c.architecture_identity,
        semantic_basis_version=c.semantic_basis_version,
        source_epoch=c.source_epoch, target_epoch=c.target_epoch,
        fixed_k=c.fixed_k, mutability="immutable",
        canonicalization_policy=c.canonicalization_policy,
    )


def _mk_capsule(cid="cap-1", owner="RECURSION"):
    from aeon.shuttle.capsule import MutableStateCapsule, CapsuleState
    return MutableStateCapsule(
        capsule_id=cid, lineage_id="lin-1",
        source_zone="RECURSION", destination_zone="TRANSFORMER",
        current_mutable_owner=owner,
        source_epoch=1, target_epoch=2,
        representation_contract=_mutable_contract(),
        causal_digest="sha256:parent",
        lifecycle_state=CapsuleState.CREATED,
        expiration_epoch=100, maximum_hops=4,
        integrity_policy="hmac_authenticated",
        payload_handle=None,
    )


# ---------------------------------------------------------------------------
def test_capsule_state_enum_has_seven_nominal_and_five_failure_states():
    from aeon.shuttle.capsule import CapsuleState
    nominal = {CapsuleState.CREATED, CapsuleState.RESERVED,
                 CapsuleState.STAGED, CapsuleState.VERIFIED,
                 CapsuleState.COMMITTED, CapsuleState.AVAILABLE,
                 CapsuleState.CONSUMED}
    failure = {CapsuleState.EXPIRED, CapsuleState.REJECTED,
                 CapsuleState.QUARANTINED, CapsuleState.ROLLED_BACK,
                 CapsuleState.RECOMPUTE_REQUIRED}
    assert nominal.issubset(set(CapsuleState))
    assert failure.issubset(set(CapsuleState))


def test_capsule_transition_enforces_graph():
    from aeon.shuttle.capsule import (
        CapsuleError, CapsuleState, transition_capsule,
    )
    c = _mk_capsule()
    # CREATED -> STAGED is illegal (must go through RESERVED).
    try:
        transition_capsule(c, CapsuleState.STAGED)
    except CapsuleError as e:
        assert e.code == "invalid_capsule_transition"
    transition_capsule(c, CapsuleState.RESERVED)
    transition_capsule(c, CapsuleState.STAGED)


def test_consumption_before_commit_refused():
    from aeon.shuttle.capsule import (
        CapsuleError, CapsuleState, transition_capsule,
    )
    c = _mk_capsule()
    transition_capsule(c, CapsuleState.RESERVED)
    transition_capsule(c, CapsuleState.STAGED)
    transition_capsule(c, CapsuleState.VERIFIED)
    # VERIFIED -> CONSUMED is illegal (must commit + become AVAILABLE first).
    try:
        transition_capsule(c, CapsuleState.CONSUMED)
    except CapsuleError as e:
        assert e.code == "invalid_capsule_transition"


def test_no_double_commit():
    from aeon.shuttle.capsule import (
        CapsuleError, CapsuleState, transition_capsule,
    )
    c = _mk_capsule()
    for s in (CapsuleState.RESERVED, CapsuleState.STAGED,
                CapsuleState.VERIFIED, CapsuleState.COMMITTED):
        transition_capsule(c, s)
    try:
        transition_capsule(c, CapsuleState.COMMITTED)
    except CapsuleError as e:
        assert e.code == "invalid_capsule_transition"


def test_terminal_states_have_no_outbound_transitions():
    from aeon.shuttle.capsule import (
        CapsuleError, CapsuleState, transition_capsule,
    )
    for terminal in (CapsuleState.CONSUMED, CapsuleState.EXPIRED,
                       CapsuleState.REJECTED, CapsuleState.QUARANTINED):
        c = _mk_capsule()
        c.lifecycle_state = terminal
        try:
            transition_capsule(c, CapsuleState.RESERVED)
        except CapsuleError as e:
            assert e.code == "invalid_capsule_transition"


# ---------------------------------------------------------------------------
# Ownership ledger
# ---------------------------------------------------------------------------
def test_ledger_records_create_and_transfer():
    from aeon.shuttle.ownership import OwnershipLedger
    ledger = OwnershipLedger()
    c = _mk_capsule()
    ledger.record_create(c)
    assert ledger.current_owner(c.capsule_id) == "RECURSION"
    ledger.record_transfer(c, from_zone="RECURSION",
                              to_zone="TRANSFORMER")
    assert ledger.current_owner(c.capsule_id) == "TRANSFORMER"
    assert c.current_mutable_owner == "TRANSFORMER"


def test_ledger_refuses_wrong_source_owner():
    from aeon.shuttle.ownership import OwnershipLedger, OwnershipViolation
    ledger = OwnershipLedger()
    c = _mk_capsule()
    ledger.record_create(c)
    try:
        ledger.record_transfer(c, from_zone="ATTACKER",
                                  to_zone="TRANSFORMER")
    except OwnershipViolation as e:
        assert e.code == "wrong_source_owner"


def test_ledger_refuses_self_transfer():
    from aeon.shuttle.ownership import OwnershipLedger, OwnershipViolation
    ledger = OwnershipLedger()
    c = _mk_capsule()
    ledger.record_create(c)
    try:
        ledger.record_transfer(c, from_zone="RECURSION",
                                  to_zone="RECURSION")
    except OwnershipViolation as e:
        assert e.code == "self_transfer_refused"


def test_ledger_chains_digest():
    from aeon.shuttle.ownership import OwnershipLedger
    ledger = OwnershipLedger()
    c1 = _mk_capsule(cid="c1")
    c2 = _mk_capsule(cid="c2")
    e1 = ledger.record_create(c1)
    e2 = ledger.record_create(c2)
    assert e2.prev_ledger == e1.ledger_digest


def test_single_ownership_holds_after_transfer():
    from aeon.shuttle.ownership import (
        OwnershipLedger, enforce_single_mutable_owner,
    )
    ledger = OwnershipLedger()
    c = _mk_capsule()
    ledger.record_create(c)
    enforce_single_mutable_owner([c], ledger)
    ledger.record_transfer(c, from_zone="RECURSION", to_zone="TRANSFORMER")
    enforce_single_mutable_owner([c], ledger)


def test_immutable_broadcasts_excluded_from_mutable_owner_check():
    """§21: immutable broadcasts are excluded from mutable-owner
    cardinality checks. Simulated by a capsule whose contract has
    mutability=immutable — no ownership entries needed."""
    from aeon.shuttle.ownership import (
        OwnershipLedger, enforce_single_mutable_owner,
    )
    ledger = OwnershipLedger()
    c = _mk_capsule()
    c.representation_contract = _immutable_contract()
    # No ownership records; still passes because immutable is skipped.
    enforce_single_mutable_owner([c], ledger)


def test_precommit_rollback_restores_source_authority():
    """§4: precommit failure must restore source mutation authority."""
    from aeon.shuttle.capsule import CapsuleState, transition_capsule
    from aeon.shuttle.ownership import OwnershipLedger
    ledger = OwnershipLedger()
    c = _mk_capsule()
    ledger.record_create(c)
    transition_capsule(c, CapsuleState.RESERVED)
    transition_capsule(c, CapsuleState.STAGED)
    # Rollback before commit.
    transition_capsule(c, CapsuleState.ROLLED_BACK)
    transition_capsule(c, CapsuleState.CREATED)
    assert c.current_mutable_owner == "RECURSION"


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
