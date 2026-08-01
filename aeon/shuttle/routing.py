"""ACIS-3 Recursion broadcast shuttle.

Given the LIVE Recursion output h_cond at a certified boundary, the
shuttle:

    1. Publishes ONE ImmutableRecursionBroadcast.
    2. Custody validates the contract.
    3. Issues TWO read-only leases (TRANSFORMER, SUBSTRATE).
    4. Resolves each lease (both return the identical live tensor).
    5. Records acknowledgement independently.
    6. Retires the broadcast after both leases are ACKNOWLEDGED
       (or immediately, if the shuttle's policy retires eagerly).

Never clones, detaches, or moves the payload tensor. The digest
computation detaches for hashing but the LIVE h_cond continues to
flow through HybridModel.forward untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Protocol, Tuple

from .audit import AcisAuditLog
from .broadcast import ImmutableRecursionBroadcast, publish_broadcast
from .contracts import (
    RepresentationContract, ContractViolation,
    TRANSFORMER as DEST_TRANSFORMER,
    SUBSTRATE as DEST_SUBSTRATE,
)
from .lease import (
    BroadcastLease, LeaseDeliveryState, acknowledge_lease,
    resolve_lease, transition,
)
from .lifecycle import BroadcastCustody
from .policy import ShuttleMode


@dataclass(frozen=True)
class BoundaryInfo:
    """One K=16 boundary hand-off. Fields hold LIVE tensor references.

    HybridModel.forward populates this when shuttle is not None; the
    shuttle then executes the ACIS lifecycle synchronously. When
    shuttle is None (the default), BoundaryInfo is never constructed.
    """
    window_index: int
    recursion_epoch: int
    token_start: int
    token_end: int
    h_cond: Any     # LIVE tensor — must not be cloned/detached by consumers
    t_w: Any
    s_w: Any
    hidden: Any
    injected: Any   # available only after transformer.inject at end of forward
    contract: RepresentationContract
    causal_parent_digest: str = "sha256:root"


class AcisBoundaryShuttle(Protocol):
    """A shuttle that HybridModel.forward calls at each boundary when
    ``shuttle`` is not None. The shuttle MUST NOT mutate any tensor
    it receives, MUST NOT raise on well-formed input under BUCKET
    mode, and MUST NOT extend the forward-path wall-clock beyond the
    documented budget."""

    mode: ShuttleMode
    audit_log: AcisAuditLog

    def on_boundary(self, info: BoundaryInfo) -> None:
        ...


@dataclass
class StandardAcisShuttle:
    """The certified ACIS shuttle. Publishes, issues leases,
    resolves, acknowledges, retires. Records the whole lifecycle to
    ``audit_log`` so recovery/replay (ACIS-6) can reconstruct the
    trace deterministically.
    """
    mode: ShuttleMode = ShuttleMode.OBSERVE
    audit_log: AcisAuditLog = field(default_factory=AcisAuditLog)
    published: List[ImmutableRecursionBroadcast] = field(default_factory=list)
    custodies: List[BroadcastCustody] = field(default_factory=list)
    lease_pairs: List[Tuple[BroadcastLease, BroadcastLease]] = field(default_factory=list)

    def on_boundary(self, info: BoundaryInfo) -> None:
        if self.mode is ShuttleMode.OFF:
            return
        # 1. Publish. Refuses on any contract mismatch.
        broadcast = publish_broadcast(
            payload=info.h_cond,
            contract=info.contract,
            recursion_epoch=info.recursion_epoch,
            boundary_index=info.window_index,
            causal_parent_digest=info.causal_parent_digest,
            expiration_epoch=info.recursion_epoch + 1024,
            integrity_level="hmac_authenticated",
        )
        self.published.append(broadcast)
        self.audit_log.append(
            kind="publish",
            boundary_index=info.window_index,
            recursion_epoch=info.recursion_epoch,
            payload_digest=broadcast.semantic_digest,
            detail={"broadcast_id": broadcast.broadcast_id})
        # 2. Custody + 3. issue lease pair.
        custody = BroadcastCustody(broadcast=broadcast)
        t_lease, s_lease = custody.issue_pair(
            route_id_transformer=f"rt-t-{info.window_index}",
            route_id_substrate=f"rt-s-{info.window_index}",
            issued_epoch=info.recursion_epoch,
            expiration_epoch=info.recursion_epoch + 1024,
            maximum_application_count=1)
        self.custodies.append(custody)
        self.lease_pairs.append((t_lease, s_lease))
        self.audit_log.append(
            kind="lease_issue",
            boundary_index=info.window_index,
            recursion_epoch=info.recursion_epoch,
            payload_digest=broadcast.semantic_digest,
            detail={"lease_ids": [t_lease.lease_id, s_lease.lease_id]})
        # 4-5. Resolve leases (both return SAME live tensor id).
        payload_t = resolve_lease(t_lease, broadcast)
        payload_s = resolve_lease(s_lease, broadcast)
        if payload_t is not payload_s:
            raise RuntimeError(
                "ACIS invariant violated: two leases must resolve to "
                "the same live tensor identity")
        # 6. Acknowledge each independently.
        acknowledge_lease(t_lease)
        acknowledge_lease(s_lease)
        self.audit_log.append(
            kind="lease_ack",
            boundary_index=info.window_index,
            recursion_epoch=info.recursion_epoch,
            payload_digest=broadcast.semantic_digest,
            detail={"lease_ids": [t_lease.lease_id, s_lease.lease_id]})
        # Release then retire.
        for l in (t_lease, s_lease):
            transition(l, LeaseDeliveryState.RELEASED)
        custody.retire()
        self.audit_log.append(
            kind="retire",
            boundary_index=info.window_index,
            recursion_epoch=info.recursion_epoch,
            payload_digest=broadcast.semantic_digest,
            detail={"broadcast_id": broadcast.broadcast_id})

    def boundaries_seen(self) -> int:
        return len(self.published)


def default_recursion_contract(
    *,
    h_rec: int,
    batch_size: int,
    model_identity: str,
    architecture_identity: str,
    recursion_epoch: int,
) -> RepresentationContract:
    """The certified representation contract for a Recursion boundary
    broadcast on the current architecture. Fixed K=16, both
    destinations permitted, immutable, contiguous_fp32_bytes
    canonicalization."""
    return RepresentationContract(
        representation_id="recursion_broadcast_v1",
        source_zone="RECURSION",
        permitted_destinations=(DEST_TRANSFORMER, DEST_SUBSTRATE),
        shape=(int(batch_size), int(h_rec)),
        dtype="float32",
        device_class="cpu",   # Detected at boundary time by callers
        model_identity=model_identity,
        architecture_identity=architecture_identity,
        semantic_basis_version=1,
        source_epoch=int(recursion_epoch),
        target_epoch=int(recursion_epoch) + 1,
        fixed_k=16,
        mutability="immutable",
        canonicalization_policy="contiguous_fp32_bytes",
    )
