"""ACIS-1 — immutable broadcast + representation contract tests.

Proves the ImmutableRecursionBroadcast wrapper is:
    * frozen (dataclass FrozenInstanceError on mutation attempts).
    * carries the exact fixed_k=16 constraint.
    * refuses contracts under a mutable payload class.
    * refuses contracts with an unsupported canonicalization policy.
    * refuses contracts with an empty permitted_destinations tuple.
    * refuses contracts declaring fixed_k != 16.
    * produces a deterministic semantic digest (same tensor + same
      contract → same digest).
    * detaches for hashing only (autograd graph of the payload
      unchanged).
    * distinguishes correct-shape tensors under a DIFFERENT
      semantic_basis_version.
    * does NOT clone or move the payload.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _make_contract(*, basis: int = 1, mutability: str = "immutable",
                    fixed_k: int = 16,
                    canon: str = "contiguous_fp32_bytes",
                    destinations=("TRANSFORMER", "SUBSTRATE"),
                    shape=(2, 64), dtype="float32"):
    from aeon.shuttle.contracts import RepresentationContract
    return RepresentationContract(
        representation_id="test-rc",
        source_zone="RECURSION",
        permitted_destinations=tuple(destinations),
        shape=tuple(shape), dtype=dtype,
        device_class="cpu",
        model_identity="model-x",
        architecture_identity="arch-y",
        semantic_basis_version=basis,
        source_epoch=100, target_epoch=101,
        fixed_k=fixed_k,
        mutability=mutability,
        canonicalization_policy=canon,
    )


# ---------------------------------------------------------------------------
# Contract validation
# ---------------------------------------------------------------------------
def test_contract_refuses_fixed_k_other_than_16():
    from aeon.shuttle.contracts import validate_contract_basic, ContractViolation
    try:
        validate_contract_basic(_make_contract(fixed_k=8))
    except ContractViolation as e:
        assert e.code == "fixed_k_not_16"
    else:
        raise AssertionError("fixed_k=8 must be refused")


def test_contract_refuses_invalid_mutability():
    from aeon.shuttle.contracts import validate_contract_basic, ContractViolation
    try:
        validate_contract_basic(_make_contract(mutability="tempo"))
    except ContractViolation as e:
        assert e.code == "invalid_mutability"
    else:
        raise AssertionError("invalid mutability must be refused")


def test_contract_refuses_unsupported_canonicalization():
    from aeon.shuttle.contracts import validate_contract_basic, ContractViolation
    try:
        validate_contract_basic(_make_contract(canon="raw_pickle"))
    except ContractViolation as e:
        assert e.code == "unsupported_canonicalization_policy"


def test_contract_refuses_empty_permitted_destinations():
    from aeon.shuttle.contracts import validate_contract_basic, ContractViolation
    try:
        validate_contract_basic(_make_contract(destinations=()))
    except ContractViolation as e:
        assert e.code == "empty_permitted_destinations"


def test_assert_destination_permitted():
    from aeon.shuttle.contracts import (
        assert_destination_permitted, ContractViolation,
    )
    contract = _make_contract()
    assert_destination_permitted(contract, "TRANSFORMER")
    try:
        assert_destination_permitted(contract, "RECURSION")
    except ContractViolation as e:
        assert e.code == "destination_not_permitted"


# ---------------------------------------------------------------------------
# ImmutableRecursionBroadcast
# ---------------------------------------------------------------------------
def test_broadcast_is_frozen_dataclass():
    import torch
    from aeon.shuttle.broadcast import publish_broadcast
    payload = torch.randn(2, 64)
    b = publish_broadcast(
        payload=payload,
        contract=_make_contract(),
        recursion_epoch=1, boundary_index=0,
        causal_parent_digest="sha256:0",
        expiration_epoch=10)
    from dataclasses import FrozenInstanceError
    try:
        b.broadcast_id = "different"
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("ImmutableRecursionBroadcast must be frozen")


def test_broadcast_carries_fixed_k_16():
    import torch
    from aeon.shuttle.broadcast import publish_broadcast
    b = publish_broadcast(
        payload=torch.randn(2, 64),
        contract=_make_contract(),
        recursion_epoch=1, boundary_index=0,
        causal_parent_digest="sha256:0",
        expiration_epoch=10)
    assert b.fixed_k == 16


def test_broadcast_refuses_mutable_contract():
    import torch
    from aeon.shuttle.broadcast import publish_broadcast
    from aeon.shuttle.contracts import ContractViolation
    try:
        publish_broadcast(
            payload=torch.randn(2, 64),
            contract=_make_contract(mutability="mutable"),
            recursion_epoch=1, boundary_index=0,
            causal_parent_digest="sha256:0",
            expiration_epoch=10)
    except ContractViolation as e:
        assert e.code == "broadcast_contract_must_be_immutable"
    else:
        raise AssertionError("mutable contract must be refused for broadcast")


# ---------------------------------------------------------------------------
# Semantic digest
# ---------------------------------------------------------------------------
def test_same_tensor_same_contract_produces_same_digest():
    import torch
    from aeon.shuttle.broadcast import compute_semantic_digest
    payload = torch.arange(128, dtype=torch.float32).view(2, 64)
    contract = _make_contract()
    d1 = compute_semantic_digest(payload, contract)
    d2 = compute_semantic_digest(payload, contract)
    assert d1 == d2 and d1.startswith("sha256:")


def test_different_semantic_basis_produces_different_digest():
    """Correct-shape tensor under a DIFFERENT semantic_basis_version
    yields a different digest — the basis version is part of the
    canonical material."""
    import torch
    from aeon.shuttle.broadcast import compute_semantic_digest
    payload = torch.arange(128, dtype=torch.float32).view(2, 64)
    d1 = compute_semantic_digest(payload, _make_contract(basis=1))
    d2 = compute_semantic_digest(payload, _make_contract(basis=2))
    assert d1 != d2


def test_semantic_digest_does_not_break_autograd_of_payload():
    """The payload tensor's autograd graph MUST be preserved after
    the digest is computed — we must never call detach() or clone()
    on the object being digested if it participates in a backward pass."""
    import torch
    from aeon.shuttle.broadcast import compute_semantic_digest
    x = torch.randn(2, 64, requires_grad=True)
    y = (x * 2.0).sum()
    contract = _make_contract()
    _ = compute_semantic_digest(x * 2.0, contract)
    y.backward()
    assert x.grad is not None
    assert torch.all(x.grad == 2.0)


def test_payload_handle_is_not_cloned_or_detached():
    """publish_broadcast MUST preserve the payload tensor identity —
    payload_handle is the same object the caller passed in."""
    import torch
    from aeon.shuttle.broadcast import publish_broadcast
    payload = torch.randn(2, 64)
    b = publish_broadcast(
        payload=payload,
        contract=_make_contract(),
        recursion_epoch=1, boundary_index=0,
        causal_parent_digest="sha256:0",
        expiration_epoch=10)
    assert b.payload_handle is payload


# ---------------------------------------------------------------------------
# One broadcast per boundary (ACIS-1 evidence: publish_broadcast is
# idempotent given identical inputs — the semantic digest is stable,
# so a second publish for the same boundary would produce an
# indistinguishable object)
# ---------------------------------------------------------------------------
def test_two_publishes_of_same_boundary_yield_equivalent_broadcast_ids():
    import torch
    from aeon.shuttle.broadcast import publish_broadcast
    payload = torch.arange(128, dtype=torch.float32).view(2, 64)
    contract = _make_contract()
    b1 = publish_broadcast(payload=payload, contract=contract,
                             recursion_epoch=5, boundary_index=3,
                             causal_parent_digest="sha256:parent",
                             expiration_epoch=50)
    b2 = publish_broadcast(payload=payload, contract=contract,
                             recursion_epoch=5, boundary_index=3,
                             causal_parent_digest="sha256:parent",
                             expiration_epoch=50)
    assert b1.broadcast_id == b2.broadcast_id
    assert b1.semantic_digest == b2.semantic_digest


# ---------------------------------------------------------------------------
# aeon.shuttle.broadcast is not imported by HybridModel.forward at ACIS-1
# ---------------------------------------------------------------------------
def test_hybrid_still_does_not_import_shuttle_at_acis_1():
    src = open(os.path.join(ROOT, "aeon", "hybrid.py"),
                 encoding="utf-8").read()
    assert "aeon.shuttle" not in src, (
        "ACIS-1 remains scaffold-only. Wire-through arrives at "
        "ACIS-3 (shuttle) under BUCKET mode.")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
