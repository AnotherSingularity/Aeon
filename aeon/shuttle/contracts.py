"""ACIS-1 representation contract + core capsule types.

A RepresentationContract describes the shape / dtype / semantic
basis / permitted destinations of one class of ACIS payload. Two
tensors with the SAME shape and dtype but DIFFERENT semantic bases
must not be considered interchangeable — the contract's
``semantic_basis_version`` is the discriminator.

Reject correct-shape tensors under the wrong semantic basis (§10).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


class ProcessingZone(str):
    """Enum-like — kept as str subclass so future zones don't break
    frozen dataclass equality checks."""


TRANSFORMER: str = "TRANSFORMER"
SUBSTRATE: str = "SUBSTRATE"
RECURSION: str = "RECURSION"

DESTINATION_TRANSFORMER = TRANSFORMER
DESTINATION_SUBSTRATE = SUBSTRATE


@dataclass(frozen=True)
class RepresentationContract:
    """One representation identity + its permitted destinations.

    ``semantic_basis_version`` is the invariant that distinguishes
    tensors with equal (shape, dtype) but different meaning. Two
    contracts with the same shape and dtype but different
    semantic_basis_version MUST NOT be treated as interchangeable
    — a shuttle that resolves a lease under the wrong contract must
    raise ``ContractViolation``.

    ``canonicalization_policy`` names the transformation applied
    before hashing the payload for a semantic digest. At ACIS-1 the
    only supported policy is ``"contiguous_fp32_bytes"``.
    """

    representation_id: str
    source_zone: str
    permitted_destinations: Tuple[str, ...]
    shape: Tuple[int, ...]
    dtype: str
    device_class: str
    model_identity: str
    architecture_identity: str
    semantic_basis_version: int
    source_epoch: int
    target_epoch: int
    fixed_k: int
    mutability: str  # "immutable" | "mutable"
    canonicalization_policy: str
    allowed_transport_transformations: Tuple[str, ...] = field(default_factory=tuple)


class ContractViolation(RuntimeError):
    """Raised on shape / dtype / semantic-basis / K / destination
    mismatches during lease resolution or shuttle publication."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def validate_contract_basic(contract: RepresentationContract) -> None:
    """Structural checks on a contract independent of any payload."""
    if contract.fixed_k != 16:
        raise ContractViolation("fixed_k_not_16", str(contract.fixed_k))
    if contract.mutability not in ("immutable", "mutable"):
        raise ContractViolation(
            "invalid_mutability", contract.mutability)
    if contract.canonicalization_policy != "contiguous_fp32_bytes":
        raise ContractViolation(
            "unsupported_canonicalization_policy",
            contract.canonicalization_policy)
    if not contract.permitted_destinations:
        raise ContractViolation(
            "empty_permitted_destinations",
            contract.representation_id)


def assert_destination_permitted(contract: RepresentationContract,
                                    destination: str) -> None:
    if destination not in contract.permitted_destinations:
        raise ContractViolation(
            "destination_not_permitted",
            f"{destination!r} not in {contract.permitted_destinations!r}")
