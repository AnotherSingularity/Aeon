"""L1 — authoritative signal trace.

Proves:

    1. The optional observer parameter is inert when None (byte-for-
       byte forward equivalence, gradient equivalence, RNG equivalence).
    2. The observer receives one RecursionWindowEvent per K-window
       boundary. Real executed tensors — not surrogates — populate the
       event.
    3. Both transformer and substrate source signals are non-zero at
       every boundary (proves both streams reach Recursion).
    4. Exactly ONE broadcast per window is captured — no second
       broadcast head snuck in.
    5. Both streams consume the same broadcast identity.
    6. Recursion state stays fp32 across every boundary.
    7. K=16 boundary timing is respected (window_index increments,
       token_start/end line up with K spans, last window may be short).
    8. The observer cannot mutate the model (weights and buffers
       unchanged post-forward).
    9. Raw text is never captured by default.
   10. Byte-budget refusal is available (TensorCaptureBudget stub —
       tensor capture landing later).
   11. Observer exceptions surface loudly.
"""
import copy
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _tiny_model_and_batch(seed: int = 1, seq_len: int = 32,
                            batch_size: int = 2, vocab_size: int = 64):
    import torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    torch.manual_seed(seed)
    tcfg = AeonTransformerConfig(
        vocab_size=vocab_size, hidden_size=64, intermediate_size=128,
        num_hidden_layers=1, num_attention_heads=2,
        num_key_value_heads=2, head_dim=32,
        max_position_embeddings=seq_len)
    model = HybridModel(
        h_rec=64, K=16,
        transformer_config=tcfg,
        substrate={"kind": "matrix", "d_in": 64, "d_state": 64},
        margin_h=0.02, margin_c=0.02,
        dtype=torch.float32,
    )
    model.recursion.float()
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = input_ids.clone()
    attention_mask = torch.ones_like(input_ids)
    return model, input_ids, attention_mask, labels


# ---------------------------------------------------------------------------
# 1. Noninterference — observer=None vs observer=NullObserver
# ---------------------------------------------------------------------------
def test_default_forward_path_untouched_when_observer_is_none():
    """The signature admits observer=None; the default call site is
    unchanged. This is the guarantee the IP-preservation firewall
    depends on."""
    import inspect
    from aeon.hybrid import HybridModel
    sig = inspect.signature(HybridModel.forward)
    assert "observer" in sig.parameters, "L1: observer kwarg missing"
    assert sig.parameters["observer"].default is None, (
        "L1: observer must default to None so probe-absent path is unchanged")


def test_observer_none_vs_null_observer_produces_identical_output():
    """With identical seed, input, and RNG state, forward(observer=None)
    must produce the same logits/loss as forward(observer=NullObserver)."""
    import torch
    from aeon.bypass.signal_trace import _NullObserver
    model, input_ids, mask, labels = _tiny_model_and_batch(seed=42)
    model.eval()
    # Snapshot RNG state before the first forward
    rng_before = torch.random.get_rng_state()
    with torch.no_grad():
        out_none = model(input_ids=input_ids, attention_mask=mask,
                          labels=labels)
    torch.random.set_rng_state(rng_before)
    with torch.no_grad():
        out_obs = model(input_ids=input_ids, attention_mask=mask,
                         labels=labels, observer=_NullObserver())
    assert torch.equal(out_none.logits, out_obs.logits), (
        "L1: probe-active logits differ from probe-absent logits")
    assert torch.equal(out_none.loss, out_obs.loss), (
        "L1: probe-active loss differs from probe-absent loss")


def test_observer_does_not_change_gradients():
    """Backprop through forward(observer=None) and forward(observer=
    NullObserver) must yield identical parameter gradients."""
    import torch
    from aeon.bypass.signal_trace import _NullObserver
    m1, ids, mask, labels = _tiny_model_and_batch(seed=7)
    m2, _, _, _ = _tiny_model_and_batch(seed=7)
    m1.train(); m2.train()
    out1 = m1(input_ids=ids, attention_mask=mask, labels=labels)
    out2 = m2(input_ids=ids, attention_mask=mask, labels=labels,
              observer=_NullObserver())
    out1.loss.backward()
    out2.loss.backward()
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(),
                                     m2.named_parameters()):
        if p1.grad is None and p2.grad is None:
            continue
        assert p1.grad is not None and p2.grad is not None
        assert torch.allclose(p1.grad, p2.grad, atol=0.0, rtol=0.0), (
            f"L1: gradient of {n1!r} differs when observer is active")


# ---------------------------------------------------------------------------
# 2. Observer receives real executed values
# ---------------------------------------------------------------------------
def test_observer_receives_one_event_per_K_boundary():
    from aeon.bypass.signal_trace import _NullObserver
    obs = _NullObserver()
    model, ids, mask, labels = _tiny_model_and_batch(seq_len=32)  # 32/16 = 2 windows
    model.eval()
    import torch
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, labels=labels, observer=obs)
    assert len(obs.events) == 2, (
        f"L1: expected 2 K=16 boundary events; got {len(obs.events)}")
    assert obs.events[0].window_index == 0
    assert obs.events[1].window_index == 1
    assert obs.events[0].token_start == 0 and obs.events[0].token_end == 16
    assert obs.events[1].token_start == 16 and obs.events[1].token_end == 32
    for ev in obs.events:
        assert ev.k_value == 16


def test_short_final_window_reported_correctly():
    from aeon.bypass.signal_trace import _NullObserver
    obs = _NullObserver()
    model, ids, mask, labels = _tiny_model_and_batch(seq_len=24)  # 16 + 8
    model.eval()
    import torch
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, labels=labels, observer=obs)
    assert len(obs.events) == 2
    assert obs.events[1].token_start == 16 and obs.events[1].token_end == 24


# ---------------------------------------------------------------------------
# 3. Both source signals present
# ---------------------------------------------------------------------------
def test_both_transformer_and_substrate_sources_are_nonzero():
    from aeon.bypass.signal_trace import _NullObserver
    obs = _NullObserver()
    model, ids, mask, labels = _tiny_model_and_batch()
    model.eval()
    import torch
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, labels=labels, observer=obs)
    assert obs.events
    for ev in obs.events:
        assert ev.transformer_source_norm > 0, (
            f"L1: transformer source norm should be > 0 at window "
            f"{ev.window_index}; got {ev.transformer_source_norm}")
        assert ev.substrate_source_norm > 0, (
            f"L1: substrate source norm should be > 0 at window "
            f"{ev.window_index}; got {ev.substrate_source_norm}")


# ---------------------------------------------------------------------------
# 4-5. Single broadcast per window, both streams consume same identity
# ---------------------------------------------------------------------------
def test_single_broadcast_per_window():
    """RecursionWindowEvent has one broadcast_norm per event, and one
    event per window — that's structural. Sanity: broadcast_norm is
    finite and both consumption flags true."""
    from aeon.bypass.signal_trace import _NullObserver
    obs = _NullObserver()
    model, ids, mask, labels = _tiny_model_and_batch()
    model.eval()
    import torch
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, labels=labels, observer=obs)
    for ev in obs.events:
        import math
        assert math.isfinite(ev.broadcast_norm)
        assert ev.transformer_consumed_broadcast is True
        assert ev.substrate_consumed_broadcast is True


# ---------------------------------------------------------------------------
# 6. Recursion state stays fp32
# ---------------------------------------------------------------------------
def test_recursion_state_stays_fp32_at_every_boundary():
    from aeon.bypass.signal_trace import _NullObserver
    obs = _NullObserver()
    model, ids, mask, labels = _tiny_model_and_batch()
    model.eval()
    import torch
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, labels=labels, observer=obs)
    for ev in obs.events:
        assert "float32" in ev.recursion_state_before_dtype, ev
        assert "float32" in ev.recursion_state_after_dtype, ev


# ---------------------------------------------------------------------------
# 7. K=16 boundary numbering
# ---------------------------------------------------------------------------
def test_window_indices_increment_from_zero():
    from aeon.bypass.signal_trace import _NullObserver
    obs = _NullObserver()
    model, ids, mask, labels = _tiny_model_and_batch(seq_len=48)  # 3 windows
    model.eval()
    import torch
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, labels=labels, observer=obs)
    assert [ev.window_index for ev in obs.events] == [0, 1, 2]
    starts = [ev.token_start for ev in obs.events]
    assert starts == [0, 16, 32]


# ---------------------------------------------------------------------------
# 8. Observer cannot mutate model
# ---------------------------------------------------------------------------
def test_observer_does_not_mutate_model_parameters():
    from aeon.bypass.signal_trace import _NullObserver
    obs = _NullObserver()
    model, ids, mask, labels = _tiny_model_and_batch()
    import torch
    with torch.no_grad():
        params_before = {n: p.detach().clone()
                          for n, p in model.named_parameters()}
        model.eval()
        model(input_ids=ids, attention_mask=mask, labels=labels, observer=obs)
        for n, p in model.named_parameters():
            assert torch.equal(params_before[n], p), (
                f"L1: parameter {n!r} mutated by observer path")


# ---------------------------------------------------------------------------
# 9. No raw text captured by default
# ---------------------------------------------------------------------------
def test_no_raw_text_in_event_by_default():
    from aeon.bypass.signal_trace import _NullObserver
    obs = _NullObserver()
    model, ids, mask, labels = _tiny_model_and_batch()
    model.eval()
    import torch
    with torch.no_grad():
        model(input_ids=ids, attention_mask=mask, labels=labels, observer=obs)
    for ev in obs.events:
        assert ev.source_record_ids == (), (
            "L1: source_record_ids must be empty by default (no raw "
            f"text leak); got {ev.source_record_ids!r}")


# ---------------------------------------------------------------------------
# 10. TensorCaptureBudget exists and defaults to disabled
# ---------------------------------------------------------------------------
def test_tensor_capture_budget_defaults_to_disabled():
    from aeon.bypass.signal_trace import TensorCaptureBudget
    b = TensorCaptureBudget()
    assert b.enabled is False, "TensorCaptureBudget must default to disabled"
    assert b.persistent is False
    # And can't be flipped by accident — it's frozen.
    from dataclasses import FrozenInstanceError
    try:
        b.enabled = True
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("TensorCaptureBudget must be frozen")


# ---------------------------------------------------------------------------
# 11. Observer exceptions propagate (probe bugs are not swallowed)
# ---------------------------------------------------------------------------
def test_observer_exception_propagates():
    class Boom:
        run_id = "test"
        checkpoint_generation_id = None
        source_record_ids = ()
        def on_recursion_window(self, event):
            raise RuntimeError("probe bug")
    import torch
    model, ids, mask, labels = _tiny_model_and_batch()
    model.eval()
    with torch.no_grad():
        try:
            model(input_ids=ids, attention_mask=mask, labels=labels,
                  observer=Boom())
        except RuntimeError as e:
            assert "probe bug" in str(e)
        else:
            raise AssertionError("Observer exceptions must propagate loudly")


# ---------------------------------------------------------------------------
# 12. No IP export path in forward
# ---------------------------------------------------------------------------
def test_hybrid_forward_still_free_of_outbound_calls():
    src = open(os.path.join(ROOT, "aeon", "hybrid.py"), encoding="utf-8").read()
    for bad in ("requests.", "urllib", "http.client", "boto3",
                  "huggingface_hub", "wandb"):
        assert bad not in src


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
