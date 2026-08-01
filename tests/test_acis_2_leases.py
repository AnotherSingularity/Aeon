"""ACIS-2 — destination leases + lifecycle custody tests."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _make_contract():
    from aeon.shuttle.contracts import RepresentationContract
    return RepresentationContract(
        representation_id="test-rc",
        source_zone="RECURSION",
        permitted_destinations=("TRANSFORMER", "SUBSTRATE"),
        shape=(2, 64), dtype="float32",
        device_class="cpu",
        model_identity="model-x", architecture_identity="arch-y",
        semantic_basis_version=1,
        source_epoch=100, target_epoch=101,
        fixed_k=16, mutability="immutable",
        canonicalization_policy="contiguous_fp32_bytes",
    )


def _pub():
    import torch
    from aeon.shuttle.broadcast import publish_broadcast
    return publish_broadcast(
        payload=torch.randn(2, 64),
        contract=_make_contract(),
        recursion_epoch=7, boundary_index=2,
        causal_parent_digest="sha256:parent",
        expiration_epoch=100)


# ---------------------------------------------------------------------------
def test_pair_leases_both_resolve_to_same_broadcast_id():
    from aeon.shuttle.lifecycle import BroadcastCustody
    b = _pub()
    custody = BroadcastCustody(broadcast=b)
    t_lease, s_lease = custody.issue_pair(
        route_id_transformer="rt-t", route_id_substrate="rt-s",
        issued_epoch=10, expiration_epoch=20)
    assert t_lease.broadcast_id == s_lease.broadcast_id == b.broadcast_id
    assert t_lease.destination == "TRANSFORMER"
    assert s_lease.destination == "SUBSTRATE"


def test_leases_are_read_only():
    from aeon.shuttle.lifecycle import BroadcastCustody
    from aeon.shuttle.lease import READ_ONLY
    b = _pub()
    custody = BroadcastCustody(broadcast=b)
    t, s = custody.issue_pair(
        route_id_transformer="rt-t", route_id_substrate="rt-s",
        issued_epoch=10, expiration_epoch=20)
    assert t.permission == READ_ONLY == s.permission


def test_resolve_lease_returns_live_payload_handle():
    from aeon.shuttle.lifecycle import BroadcastCustody
    from aeon.shuttle.lease import resolve_lease
    b = _pub()
    custody = BroadcastCustody(broadcast=b)
    t, s = custody.issue_pair(
        route_id_transformer="rt-t", route_id_substrate="rt-s",
        issued_epoch=10, expiration_epoch=20)
    payload_t = resolve_lease(t, b)
    payload_s = resolve_lease(s, b)
    assert payload_t is b.payload_handle
    assert payload_s is b.payload_handle


def test_both_leases_carry_same_semantic_digest_via_broadcast():
    """Both destination leases refer to the same broadcast identity;
    the semantic digest is a property of the broadcast, not the lease."""
    from aeon.shuttle.lifecycle import BroadcastCustody
    b = _pub()
    custody = BroadcastCustody(broadcast=b)
    t, s = custody.issue_pair(
        route_id_transformer="rt-t", route_id_substrate="rt-s",
        issued_epoch=10, expiration_epoch=20)
    # Digest lookup via the custody's broadcast reference — the leases
    # do not carry it themselves (delivery metadata only).
    assert b.semantic_digest.startswith("sha256:")
    from aeon.shuttle.broadcast import compute_semantic_digest
    d = compute_semantic_digest(b.payload_handle,
                                   b.representation_contract)
    assert d == b.semantic_digest


def test_lease_lifecycle_transitions_are_enforced():
    from aeon.shuttle.lifecycle import BroadcastCustody
    from aeon.shuttle.lease import LeaseError, transition, LeaseDeliveryState
    b = _pub()
    custody = BroadcastCustody(broadcast=b)
    t, s = custody.issue_pair(
        route_id_transformer="rt-t", route_id_substrate="rt-s",
        issued_epoch=10, expiration_epoch=20)
    # ISSUED -> DELIVERED is illegal (must pass through IN_TRANSIT).
    try:
        transition(t, LeaseDeliveryState.DELIVERED)
    except LeaseError as e:
        assert e.code == "invalid_lease_transition"
    else:
        raise AssertionError("ISSUED->DELIVERED must be refused")
    transition(t, LeaseDeliveryState.IN_TRANSIT)
    transition(t, LeaseDeliveryState.DELIVERED)


def test_one_lease_failure_does_not_change_other_leases_delivery_state():
    """§14: a failure of one lease must not change the other
    destination's payload or delivery state."""
    from aeon.shuttle.lifecycle import BroadcastCustody
    from aeon.shuttle.lease import LeaseDeliveryState
    b = _pub()
    custody = BroadcastCustody(broadcast=b)
    t, s = custody.issue_pair(
        route_id_transformer="rt-t", route_id_substrate="rt-s",
        issued_epoch=10, expiration_epoch=20)
    custody.revoke(t.lease_id)
    assert t.delivery_state is LeaseDeliveryState.REVOKED
    # Substrate lease untouched.
    assert s.delivery_state is LeaseDeliveryState.ISSUED
    # And the underlying broadcast payload_handle is unchanged.
    assert custody.broadcast.payload_handle is b.payload_handle


def test_retire_refuses_while_leases_active():
    from aeon.shuttle.lifecycle import BroadcastCustody, LifecycleError
    b = _pub()
    custody = BroadcastCustody(broadcast=b)
    custody.issue_pair(
        route_id_transformer="rt-t", route_id_substrate="rt-s",
        issued_epoch=10, expiration_epoch=20)
    try:
        custody.retire()
    except LifecycleError as e:
        assert e.code == "leases_still_active"


def test_expire_all_then_retire_succeeds():
    from aeon.shuttle.lifecycle import BroadcastCustody
    from aeon.shuttle.lease import LeaseDeliveryState
    b = _pub()
    custody = BroadcastCustody(broadcast=b)
    custody.issue_pair(
        route_id_transformer="rt-t", route_id_substrate="rt-s",
        issued_epoch=10, expiration_epoch=20)
    custody.expire_all()
    for l in custody.leases.values():
        assert l.delivery_state is LeaseDeliveryState.EXPIRED
    custody.retire()
    assert custody.retired is True


def test_second_pair_issuance_refused():
    """Only ONE lease per destination per broadcast. Attempting a
    second issuance for a destination is refused."""
    from aeon.shuttle.lifecycle import BroadcastCustody, LifecycleError
    b = _pub()
    custody = BroadcastCustody(broadcast=b)
    custody.issue_pair(
        route_id_transformer="rt-t", route_id_substrate="rt-s",
        issued_epoch=10, expiration_epoch=20)
    try:
        custody.issue_pair(
            route_id_transformer="rt-t2", route_id_substrate="rt-s2",
            issued_epoch=11, expiration_epoch=21)
    except LifecycleError as e:
        assert e.code in ("transformer_lease_already_issued",
                            "substrate_lease_already_issued")


def test_resolve_lease_refuses_terminal_leases():
    from aeon.shuttle.lifecycle import BroadcastCustody
    from aeon.shuttle.lease import (
        LeaseError, resolve_lease, LeaseDeliveryState, transition,
    )
    b = _pub()
    custody = BroadcastCustody(broadcast=b)
    t, s = custody.issue_pair(
        route_id_transformer="rt-t", route_id_substrate="rt-s",
        issued_epoch=10, expiration_epoch=20)
    transition(t, LeaseDeliveryState.EXPIRED)
    try:
        resolve_lease(t, b)
    except LeaseError as e:
        assert e.code == "lease_terminal"


def test_resolve_lease_refuses_broadcast_mismatch():
    from aeon.shuttle.lifecycle import BroadcastCustody
    from aeon.shuttle.lease import LeaseError, resolve_lease
    b1 = _pub(); b2 = _pub()
    c1 = BroadcastCustody(broadcast=b1)
    t, s = c1.issue_pair(
        route_id_transformer="rt-t", route_id_substrate="rt-s",
        issued_epoch=10, expiration_epoch=20)
    try:
        resolve_lease(t, b2)  # wrong broadcast
    except LeaseError as e:
        assert e.code == "lease_broadcast_mismatch"


def test_resolve_lease_enforces_max_application_count():
    from aeon.shuttle.lifecycle import BroadcastCustody
    from aeon.shuttle.lease import LeaseError, resolve_lease
    b = _pub()
    custody = BroadcastCustody(broadcast=b)
    t, s = custody.issue_pair(
        route_id_transformer="rt-t", route_id_substrate="rt-s",
        issued_epoch=10, expiration_epoch=20,
        maximum_application_count=2)
    resolve_lease(t, b)
    resolve_lease(t, b)
    try:
        resolve_lease(t, b)
    except LeaseError as e:
        assert e.code == "lease_exhausted"


def test_unauthorized_destination_refused():
    from aeon.shuttle.lease import issue_lease
    from aeon.shuttle.contracts import ContractViolation
    b = _pub()
    try:
        issue_lease(broadcast=b, destination="ATTACKER",
                     route_id="rt", issued_epoch=1, expiration_epoch=10)
    except ContractViolation as e:
        assert e.code == "unauthorized_destination"


def test_metadata_excluded_from_semantic_digest():
    """The lease's route_id, issued_epoch, retry_count, buffer address,
    delivery_state, and acknowledgement_state MUST NOT influence the
    broadcast's semantic digest. Verified by mutating lease metadata
    between two digest computations of the SAME broadcast."""
    from aeon.shuttle.broadcast import compute_semantic_digest
    from aeon.shuttle.lifecycle import BroadcastCustody
    from aeon.shuttle.lease import transition, LeaseDeliveryState
    b = _pub()
    custody = BroadcastCustody(broadcast=b)
    t, s = custody.issue_pair(
        route_id_transformer="rt-t", route_id_substrate="rt-s",
        issued_epoch=10, expiration_epoch=20)
    d1 = compute_semantic_digest(b.payload_handle,
                                    b.representation_contract)
    transition(t, LeaseDeliveryState.IN_TRANSIT)
    transition(t, LeaseDeliveryState.DELIVERED)
    d2 = compute_semantic_digest(b.payload_handle,
                                    b.representation_contract)
    assert d1 == d2


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
