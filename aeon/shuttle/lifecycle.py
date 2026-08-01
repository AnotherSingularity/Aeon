"""ACIS-2 lifecycle custodian.

Owns one immutable broadcast and its two leases; may issue, revoke,
retire, and expire. Never mutates the semantic payload — only the
lease-delivery metadata and the retirement flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .broadcast import ImmutableRecursionBroadcast
from .lease import (
    BroadcastLease, LeaseDeliveryState, LeaseError,
    issue_lease, transition,
)
from .contracts import (
    TRANSFORMER as DEST_TRANSFORMER,
    SUBSTRATE as DEST_SUBSTRATE,
    ContractViolation,
)


class LifecycleError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass
class BroadcastCustody:
    broadcast: ImmutableRecursionBroadcast
    leases: Dict[str, BroadcastLease] = field(default_factory=dict)
    retired: bool = False

    def issue_pair(
        self,
        *,
        route_id_transformer: str,
        route_id_substrate: str,
        issued_epoch: int,
        expiration_epoch: int,
        maximum_application_count: int = 1,
    ) -> "tuple[BroadcastLease, BroadcastLease]":
        """Issue both leases for this broadcast atomically. Either both
        leases are added or neither is."""
        if self.retired:
            raise LifecycleError("broadcast_retired", self.broadcast.broadcast_id)
        if DEST_TRANSFORMER in [l.destination for l in self.leases.values()]:
            raise LifecycleError("transformer_lease_already_issued")
        if DEST_SUBSTRATE in [l.destination for l in self.leases.values()]:
            raise LifecycleError("substrate_lease_already_issued")
        t_lease = issue_lease(
            broadcast=self.broadcast, destination=DEST_TRANSFORMER,
            route_id=route_id_transformer,
            issued_epoch=issued_epoch, expiration_epoch=expiration_epoch,
            maximum_application_count=maximum_application_count)
        s_lease = issue_lease(
            broadcast=self.broadcast, destination=DEST_SUBSTRATE,
            route_id=route_id_substrate,
            issued_epoch=issued_epoch, expiration_epoch=expiration_epoch,
            maximum_application_count=maximum_application_count)
        self.leases[t_lease.lease_id] = t_lease
        self.leases[s_lease.lease_id] = s_lease
        return t_lease, s_lease

    def revoke(self, lease_id: str) -> BroadcastLease:
        if lease_id not in self.leases:
            raise LifecycleError("unknown_lease", lease_id)
        lease = self.leases[lease_id]
        transition(lease, LeaseDeliveryState.REVOKED)
        return lease

    def expire_all(self) -> None:
        for lease in self.leases.values():
            if not lease.is_terminal():
                transition(lease, LeaseDeliveryState.EXPIRED)

    def retire(self) -> None:
        """Retire the broadcast custody. Refuses if any lease is
        still active (§4 lifecycle custodian may not mutate the
        semantic payload; retirement finalises the transport)."""
        for lease in self.leases.values():
            if not lease.is_terminal():
                raise LifecycleError(
                    "leases_still_active",
                    f"{lease.lease_id} in {lease.delivery_state.value}")
        self.retired = True

    def other_destination_state(self, my_destination: str) -> Optional[
            LeaseDeliveryState]:
        """One destination's failure must not change the other's
        payload — §14 requires that leases fail independently. This
        helper is used by tests to prove that after revoking one
        lease, the other lease's delivery_state is untouched."""
        for lease in self.leases.values():
            if lease.destination != my_destination:
                return lease.delivery_state
        return None
