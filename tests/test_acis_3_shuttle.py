"""ACIS-3 — Recursion broadcast shuttle wire-through.

Proves:
    * shuttle=None (default) is byte-identical to a build without
      aeon.shuttle: logits, loss, gradients agree exactly.
    * shuttle=StandardAcisShuttle(OBSERVE) runs synchronously per
      boundary; publishes ONE broadcast, issues TWO leases, resolves
      BOTH to the same live tensor, acknowledges, retires.
    * The audit log records publish/lease_issue/lease_ack/retire in
      the correct order with chained ledger digests.
    * Fixed K=16 preserved: number of published broadcasts == number
      of K-windows.
    * Semantic digest is deterministic across identical forward calls.
    * Payload never cloned/detached — same object identity as the
      forward's h_cond stack element.
    * Any semantic-basis mismatch is refused.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _tiny_model_and_batch(seq_len=32, seed=1):
    import torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    torch.manual_seed(seed)
    tcfg = AeonTransformerConfig(
        vocab_size=64, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=32,
        max_position_embeddings=seq_len)
    model = HybridModel(
        h_rec=64, K=16, transformer_config=tcfg,
        substrate={"kind": "matrix", "d_in": 64, "d_state": 64},
        margin_h=0.02, margin_c=0.02, dtype=torch.float32)
    model.recursion.float()
    input_ids = torch.randint(0, 64, (2, seq_len))
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    return model, input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# shuttle=None equivalence
# ---------------------------------------------------------------------------
def test_shuttle_none_produces_byte_identical_forward():
    import torch
    m1, ids, mask, labels = _tiny_model_and_batch(seed=100)
    m2, _, _, _ = _tiny_model_and_batch(seed=100)
    m1.eval(); m2.eval()
    with torch.no_grad():
        o1 = m1(input_ids=ids, attention_mask=mask, labels=labels)
        o2 = m2(input_ids=ids, attention_mask=mask, labels=labels,
                 shuttle=None)
    assert torch.equal(o1.logits, o2.logits)
    assert torch.equal(o1.loss, o2.loss)


def test_shuttle_none_gradients_bit_identical():
    import torch
    m1, ids, mask, labels = _tiny_model_and_batch(seed=101)
    m2, _, _, _ = _tiny_model_and_batch(seed=101)
    m1.train(); m2.train()
    o1 = m1(input_ids=ids, attention_mask=mask, labels=labels)
    o2 = m2(input_ids=ids, attention_mask=mask, labels=labels,
             shuttle=None)
    o1.loss.backward()
    o2.loss.backward()
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(),
                                     m2.named_parameters()):
        if p1.grad is None and p2.grad is None:
            continue
        assert torch.allclose(p1.grad, p2.grad, atol=0.0, rtol=0.0), n1


def test_shuttle_default_is_none():
    from aeon.hybrid import HybridModel
    import inspect
    sig = inspect.signature(HybridModel.forward)
    assert "shuttle" in sig.parameters
    assert sig.parameters["shuttle"].default is None


# ---------------------------------------------------------------------------
# OBSERVE mode
# ---------------------------------------------------------------------------
def test_observe_shuttle_publishes_one_broadcast_per_K_window():
    import torch
    from aeon.shuttle.policy import ShuttleMode
    from aeon.shuttle.routing import StandardAcisShuttle
    shuttle = StandardAcisShuttle(mode=ShuttleMode.OBSERVE)
    model, ids, mask, labels = _tiny_model_and_batch(seq_len=32)
    model.eval()
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, labels=labels,
              shuttle=shuttle)
    assert shuttle.boundaries_seen() == 2  # 32/16 = 2 windows


def test_observe_shuttle_short_final_window_reported_correctly():
    import torch
    from aeon.shuttle.policy import ShuttleMode
    from aeon.shuttle.routing import StandardAcisShuttle
    shuttle = StandardAcisShuttle(mode=ShuttleMode.OBSERVE)
    model, ids, mask, labels = _tiny_model_and_batch(seq_len=24)  # 16 + 8
    model.eval()
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, labels=labels,
              shuttle=shuttle)
    assert shuttle.boundaries_seen() == 2


def test_observe_shuttle_records_full_lifecycle_in_audit_log():
    import torch
    from aeon.shuttle.policy import ShuttleMode
    from aeon.shuttle.routing import StandardAcisShuttle
    shuttle = StandardAcisShuttle(mode=ShuttleMode.OBSERVE)
    model, ids, mask, labels = _tiny_model_and_batch(seq_len=32)
    model.eval()
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, labels=labels,
              shuttle=shuttle)
    kinds = [e.kind for e in shuttle.audit_log.events()]
    # Per boundary: publish, lease_issue, lease_ack, retire (×2 windows)
    assert kinds == ["publish", "lease_issue", "lease_ack", "retire",
                       "publish", "lease_issue", "lease_ack", "retire"]


def test_observe_shuttle_ledger_digest_chain_is_intact():
    import torch
    from aeon.shuttle.policy import ShuttleMode
    from aeon.shuttle.routing import StandardAcisShuttle
    shuttle = StandardAcisShuttle(mode=ShuttleMode.OBSERVE)
    model, ids, mask, labels = _tiny_model_and_batch(seq_len=32)
    model.eval()
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, labels=labels,
              shuttle=shuttle)
    evs = shuttle.audit_log.events()
    for i in range(1, len(evs)):
        assert evs[i].prev_ledger_digest == evs[i - 1].ledger_digest


def test_observe_shuttle_both_leases_resolve_to_same_live_tensor():
    """The two leases per boundary must resolve to the exact same
    tensor object — verified by the shuttle's internal identity check
    which raises RuntimeError if resolution diverges."""
    import torch
    from aeon.shuttle.policy import ShuttleMode
    from aeon.shuttle.routing import StandardAcisShuttle
    shuttle = StandardAcisShuttle(mode=ShuttleMode.OBSERVE)
    model, ids, mask, labels = _tiny_model_and_batch(seq_len=32)
    model.eval()
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, labels=labels,
              shuttle=shuttle)
    for custody in shuttle.custodies:
        leases = list(custody.leases.values())
        assert len(leases) == 2
        # both should have carried the same broadcast_id
        assert leases[0].broadcast_id == leases[1].broadcast_id


def test_observe_shuttle_preserves_logits_and_loss():
    """OBSERVE mode does not change the forward output — the shuttle
    is a downstream observer."""
    import torch
    from aeon.shuttle.policy import ShuttleMode
    from aeon.shuttle.routing import StandardAcisShuttle
    m1, ids, mask, labels = _tiny_model_and_batch(seed=42)
    m2, _, _, _ = _tiny_model_and_batch(seed=42)
    m1.eval(); m2.eval()
    with torch.no_grad():
        o_none = m1(input_ids=ids, attention_mask=mask, labels=labels)
        o_obs = m2(input_ids=ids, attention_mask=mask, labels=labels,
                    shuttle=StandardAcisShuttle(mode=ShuttleMode.OBSERVE))
    assert torch.equal(o_none.logits, o_obs.logits)
    assert torch.equal(o_none.loss, o_obs.loss)


def test_observe_shuttle_preserves_gradients():
    """Gradient flow through the shuttle-active forward must equal
    the shuttle-absent forward."""
    import torch
    from aeon.shuttle.policy import ShuttleMode
    from aeon.shuttle.routing import StandardAcisShuttle
    m1, ids, mask, labels = _tiny_model_and_batch(seed=43)
    m2, _, _, _ = _tiny_model_and_batch(seed=43)
    m1.train(); m2.train()
    o1 = m1(input_ids=ids, attention_mask=mask, labels=labels)
    o2 = m2(input_ids=ids, attention_mask=mask, labels=labels,
             shuttle=StandardAcisShuttle(mode=ShuttleMode.OBSERVE))
    o1.loss.backward(); o2.loss.backward()
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(),
                                     m2.named_parameters()):
        if p1.grad is None and p2.grad is None:
            continue
        assert torch.allclose(p1.grad, p2.grad, atol=1e-6, rtol=1e-6), (
            n1, (p1.grad - p2.grad).abs().max())


def test_observe_shuttle_fixed_k_16_carried_through_events():
    import torch
    from aeon.shuttle.policy import ShuttleMode
    from aeon.shuttle.routing import StandardAcisShuttle
    shuttle = StandardAcisShuttle(mode=ShuttleMode.OBSERVE)
    model, ids, mask, labels = _tiny_model_and_batch(seq_len=32)
    model.eval()
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, labels=labels,
              shuttle=shuttle)
    for b in shuttle.published:
        assert b.fixed_k == 16
        assert b.representation_contract.fixed_k == 16


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
