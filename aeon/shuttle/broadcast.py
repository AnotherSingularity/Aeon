"""ACIS-1 immutable Recursion broadcast.

An ImmutableRecursionBroadcast wraps ONE Recursion output boundary
in metadata + a payload handle. The handle IS the live tensor —
never a clone/detach — so autograd, device, and dtype survive
untouched. The semantic digest is computed once at publish, over
the canonical bytes, and never re-derived.

At OBSERVE mode this object is created; at OFF mode it is not.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Optional

from .contracts import RepresentationContract, ContractViolation


class SemanticDigestError(RuntimeError):
    pass


@dataclass(frozen=True)
class ImmutableRecursionBroadcast:
    """One certified Recursion broadcast identity.

    ``payload_handle`` refers to the live h_cond tensor. No clone/
    detach is made at construction. Downstream code that needs raw
    bytes for hashing MUST use ``compute_semantic_digest`` which
    detaches for hashing only — the payload_handle itself remains
    the graph-connected tensor.
    """

    broadcast_id: str
    recursion_epoch: int
    fixed_k: int
    payload_handle: Any
    representation_contract: RepresentationContract
    schema_version: int
    semantic_digest: str
    causal_parent_digest: str
    model_identity: str
    architecture_identity: str
    expiration_epoch: int
    integrity_level: str


def compute_semantic_digest(payload: Any,
                              contract: RepresentationContract) -> str:
    """Return a stable SHA-256 over the canonical bytes of ``payload``.

    Uses ``contract.canonicalization_policy`` to select the byte
    representation. Detaches for hashing — the payload's graph
    connection is NOT altered by this call.
    """
    if contract.canonicalization_policy != "contiguous_fp32_bytes":
        raise SemanticDigestError(
            "unsupported_canonicalization_policy: "
            f"{contract.canonicalization_policy!r}")
    try:
        import torch
    except Exception as e:
        raise SemanticDigestError(f"torch_unavailable: {e}") from e
    if not isinstance(payload, torch.Tensor):
        raise SemanticDigestError(
            f"payload_not_tensor: {type(payload).__name__}")
    with torch.no_grad():
        canon = payload.detach().to(dtype=torch.float32).contiguous()
    material = (
        f"basis={contract.semantic_basis_version}|"
        f"shape={tuple(int(d) for d in canon.shape)}|"
        f"dtype=float32|"
    ).encode("utf-8") + canon.cpu().numpy().tobytes()
    return "sha256:" + hashlib.sha256(material).hexdigest()


def publish_broadcast(
    *,
    payload: Any,
    contract: RepresentationContract,
    recursion_epoch: int,
    boundary_index: int,
    causal_parent_digest: str,
    expiration_epoch: int,
    integrity_level: str = "hmac_authenticated",
    schema_version: int = 1,
) -> ImmutableRecursionBroadcast:
    """Construct one immutable broadcast for the current boundary.

    Refuses under any contract-level violation. Never modifies the
    payload tensor.
    """
    # Basic contract sanity.
    from .contracts import validate_contract_basic
    validate_contract_basic(contract)
    if contract.mutability != "immutable":
        raise ContractViolation(
            "broadcast_contract_must_be_immutable",
            contract.mutability)
    if contract.fixed_k != 16:
        raise ContractViolation("fixed_k_not_16", str(contract.fixed_k))
    # Semantic digest computed once from the live tensor.
    digest = compute_semantic_digest(payload, contract)
    broadcast_id = (
        f"acis-b-{recursion_epoch:010d}-{boundary_index:06d}-"
        + digest[:16]
    )
    return ImmutableRecursionBroadcast(
        broadcast_id=broadcast_id,
        recursion_epoch=int(recursion_epoch),
        fixed_k=int(contract.fixed_k),
        payload_handle=payload,
        representation_contract=contract,
        schema_version=schema_version,
        semantic_digest=digest,
        causal_parent_digest=causal_parent_digest,
        model_identity=contract.model_identity,
        architecture_identity=contract.architecture_identity,
        expiration_epoch=int(expiration_epoch),
        integrity_level=integrity_level,
    )
