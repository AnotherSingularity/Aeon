"""ACIS-2 destination read leases + lifecycle states.

Each Recursion broadcast admits TWO leases: one for the transformer
destination, one for the substrate destination. Both resolve to the
same underlying payload_handle, same semantic_digest, same causal
parent — only per-lease delivery metadata differs (§3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from .contracts import (
    ContractViolation, assert_destination_permitted,
    RepresentationContract,
    TRANSFORMER as DEST_TRANSFORMER,
    SUBSTRATE as DEST_SUBSTRATE,
)


class LeaseDeliveryState(str, Enum):
    ISSUED = "ISSUED"
    IN_TRANSIT = "IN_TRANSIT"
    DELIVERED = "DELIVERED"
    READ = "READ"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RELEASED = "RELEASED"

    # Failure states
    EXPIRED = "EXPIRED"
    REVOKED = "REVOKED"
    DELIVERY_FAILED = "DELIVERY_FAILED"
    INTEGRITY_REJECTED = "INTEGRITY_REJECTED"
    DESTINATION_REJECTED = "DESTINATION_REJECTED"


READ_ONLY = "READ_ONLY"


class LeaseError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass
class BroadcastLease:
    lease_id: str
    broadcast_id: str
    destination: str
    permission: str
    route_id: str
    issued_epoch: int
    expiration_epoch: int
    delivery_state: LeaseDeliveryState
    acknowledgement_state: str
    integrity_reference: str
    application_count: int
    maximum_application_count: int

    def is_terminal(self) -> bool:
        return self.delivery_state in (
            LeaseDeliveryState.RELEASED,
            LeaseDeliveryState.EXPIRED,
            LeaseDeliveryState.REVOKED,
            LeaseDeliveryState.DELIVERY_FAILED,
            LeaseDeliveryState.INTEGRITY_REJECTED,
            LeaseDeliveryState.DESTINATION_REJECTED,
        )


def _valid_transitions() -> "dict[LeaseDeliveryState, set[LeaseDeliveryState]]":
    S = LeaseDeliveryState
    return {
        S.ISSUED:       {S.IN_TRANSIT, S.EXPIRED, S.REVOKED,
                          S.DESTINATION_REJECTED, S.INTEGRITY_REJECTED},
        S.IN_TRANSIT:   {S.DELIVERED, S.DELIVERY_FAILED, S.EXPIRED, S.REVOKED},
        S.DELIVERED:    {S.READ, S.EXPIRED, S.REVOKED,
                          S.DESTINATION_REJECTED, S.INTEGRITY_REJECTED},
        S.READ:         {S.ACKNOWLEDGED, S.EXPIRED, S.REVOKED},
        S.ACKNOWLEDGED: {S.RELEASED, S.EXPIRED},
        S.RELEASED:     set(),
        S.EXPIRED:      set(),
        S.REVOKED:      set(),
        S.DELIVERY_FAILED: {S.EXPIRED, S.REVOKED},
        S.INTEGRITY_REJECTED: set(),
        S.DESTINATION_REJECTED: set(),
    }


def transition(lease: BroadcastLease,
                 to: LeaseDeliveryState) -> BroadcastLease:
    """Mutate the lease's delivery_state under the allowed transition
    graph. Refuses illegal transitions and terminal-to-anything moves."""
    allowed = _valid_transitions().get(lease.delivery_state, set())
    if to not in allowed:
        raise LeaseError(
            "invalid_lease_transition",
            f"{lease.delivery_state.value} -> {to.value}")
    lease.delivery_state = to
    return lease


def issue_lease(
    *,
    broadcast,             # ImmutableRecursionBroadcast
    destination: str,
    route_id: str,
    issued_epoch: int,
    expiration_epoch: int,
    maximum_application_count: int = 1,
    integrity_reference: str = "hmac_authenticated",
) -> BroadcastLease:
    """Issue ONE lease for one destination. Refuses:
        * permission other than READ_ONLY (leases are read-only).
        * destination not permitted by the broadcast's contract.
        * expiration <= issued epoch.
    """
    if destination not in (DEST_TRANSFORMER, DEST_SUBSTRATE):
        raise ContractViolation(
            "unauthorized_destination",
            f"{destination!r} not in "
            f"{(DEST_TRANSFORMER, DEST_SUBSTRATE)!r}")
    assert_destination_permitted(broadcast.representation_contract,
                                    destination)
    if expiration_epoch <= issued_epoch:
        raise LeaseError("expiration_not_after_issue",
                           f"issued={issued_epoch} exp={expiration_epoch}")
    lease_id = (
        f"acis-l-{broadcast.recursion_epoch:010d}-"
        f"{broadcast.broadcast_id[-16:]}-{destination.lower()}"
    )
    return BroadcastLease(
        lease_id=lease_id,
        broadcast_id=broadcast.broadcast_id,
        destination=destination,
        permission=READ_ONLY,
        route_id=route_id,
        issued_epoch=int(issued_epoch),
        expiration_epoch=int(expiration_epoch),
        delivery_state=LeaseDeliveryState.ISSUED,
        acknowledgement_state="PENDING",
        integrity_reference=integrity_reference,
        application_count=0,
        maximum_application_count=int(maximum_application_count),
    )


def resolve_lease(lease: BroadcastLease, broadcast) -> Any:
    """Return the LIVE payload_handle for a valid lease. Refuses
    expired/revoked/terminal/exceeded leases; increments
    application_count. NEVER clones, detaches, or moves the tensor.

    The tests prove that resolving a lease is bit-identical to the
    payload the broadcast holds — leases carry delivery metadata
    only, not semantic transformations.
    """
    if lease.broadcast_id != broadcast.broadcast_id:
        raise LeaseError("lease_broadcast_mismatch",
                           f"{lease.broadcast_id} vs {broadcast.broadcast_id}")
    if lease.permission != READ_ONLY:
        raise LeaseError("lease_not_read_only", lease.permission)
    if lease.is_terminal():
        raise LeaseError("lease_terminal", lease.delivery_state.value)
    if lease.application_count >= lease.maximum_application_count:
        raise LeaseError("lease_exhausted",
                           f"count={lease.application_count}")
    lease.application_count += 1
    return broadcast.payload_handle


def acknowledge_lease(lease: BroadcastLease) -> BroadcastLease:
    """Move ISSUED/DELIVERED/READ to ACKNOWLEDGED (through required
    intermediate states). Convenience wrapper for tests."""
    S = LeaseDeliveryState
    path = {
        S.ISSUED: (S.IN_TRANSIT, S.DELIVERED, S.READ, S.ACKNOWLEDGED),
        S.IN_TRANSIT: (S.DELIVERED, S.READ, S.ACKNOWLEDGED),
        S.DELIVERED: (S.READ, S.ACKNOWLEDGED),
        S.READ: (S.ACKNOWLEDGED,),
    }.get(lease.delivery_state)
    if path is None:
        raise LeaseError("cannot_acknowledge",
                           lease.delivery_state.value)
    for step in path:
        transition(lease, step)
    return lease
