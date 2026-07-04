"""
Substrate adaptive-feedback sanity tests (the closed-loop control extension).

Covers the extension's invariants:
  1. load metric L(t) is bounded
  2. gate g(L) is differentiable and stays in [0, 1]
  3. gate fully OFF reduces exactly to the pre-extension readout (+ bounded)
  4. gate fully ON stays bounded (stressed mode is bound-preserving) + certificate holds
  5. certificate holds gate-off, gate-on, and mid-transition
  6. W_stressed produces output measurably different from the normal path

Requires torch; skips cleanly otherwise. Run: python tests/test_feedback.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _have_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _cell(seed=0, **kw):
    import torch
    from aeon.substrate.matrix_cell import MatrixStateCell
    torch.manual_seed(seed)
    # base-path weights are created before the feedback controller, so two cells
    # with the same seed share identical base weights regardless of feedback.
    return MatrixStateCell(d_in=16, d_state=24, n_head=2, head_size=12, **kw)


def _run_cell(cell, steps=25, B=2, seed=1):
    import torch
    torch.manual_seed(seed)
    cell.reset(B)
    outs = [cell.step(torch.randn(B, cell.d_in)) for _ in range(steps)]
    return torch.stack(outs)


def _tiny_model(seed=0, **subcfg):
    import torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    torch.manual_seed(seed)
    tcfg = AeonTransformerConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=64, tie_word_embeddings=True)
    sub = {"kind": "matrix", "d_in": 24, "d_state": 24, "n_head": 2, "head_size": 12}
    sub.update(subcfg)
    m = HybridModel(h_rec=24, K=4, transformer_config=tcfg, substrate=sub,
                    freeze_backbone=False, use_embedding_input=True, dtype=torch.float32)
    m.recursion.float()
    return m


def test_load_metric_bounded():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    cell = _cell()
    cell.reset(2)
    for _ in range(30):
        cell.step(torch.randn(2, cell.d_in))
        L = cell.load()
        assert L is not None and torch.isfinite(L).all()
        # L is an EWMA of mean|Δreadout|, readout ∈ (-1,1) ⇒ L ∈ [0, 2·bound]
        assert (L >= 0).all() and (L <= 2.0 * cell.output_bound + 1e-4).all(), L


def test_gate_range_and_differentiable():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.substrate.feedback import AdaptiveFeedbackController
    fb = AdaptiveFeedbackController(d_state=8, threshold_init=0.1, alpha_init=6.0)
    fb.reset()
    fb(torch.randn(3, 8).tanh())
    out = fb(torch.randn(3, 8).tanh().requires_grad_(True))
    g = fb.gate()
    assert g is not None and (g >= 0).all() and (g <= 1).all(), g   # valid range
    out.sum().backward()                                            # differentiable
    assert fb.gate_alpha.grad is not None and torch.isfinite(fb.gate_alpha.grad)
    assert fb.gate_threshold.grad is not None and torch.isfinite(fb.gate_threshold.grad)


def test_gate_off_reduces_to_base():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    fb_cell = _cell(seed=0, adaptive_feedback=True)
    base_cell = _cell(seed=0, adaptive_feedback=False)   # identical base-path weights
    fb_cell.feedback.gate_threshold.data.fill_(1e9)      # force g ≡ 0
    out_fb = _run_cell(fb_cell)
    out_base = _run_cell(base_cell)
    assert torch.allclose(out_fb, out_base, atol=1e-6), (out_fb - out_base).abs().max()
    assert out_fb.abs().max() <= fb_cell.output_bound + 1e-5
    assert (fb_cell.gate() == 0).all()


def test_gate_on_bounded_and_differs():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    fb_cell = _cell(seed=0, adaptive_feedback=True)
    base_cell = _cell(seed=0, adaptive_feedback=False)
    fb_cell.feedback.gate_threshold.data.fill_(-1e9)     # force g ≡ 1 (stressed mode)
    out_fb = _run_cell(fb_cell)
    out_base = _run_cell(base_cell)
    # stressed output stays bounded (bound-preserving by construction) ...
    assert out_fb.abs().max() <= fb_cell.output_bound + 1e-5
    assert (fb_cell.gate() >= 1.0 - 1e-6).all()
    # ... and is MEASURABLY different from the normal path
    assert (out_fb - out_base).abs().max() > 1e-3, (out_fb - out_base).abs().max()


def test_certificate_holds_all_modes():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    for label, thr in (("off", 1e9), ("on", -1e9), ("transition", None)):
        m = _tiny_model(seed=0)
        if thr is not None:
            m.substrate.feedback.gate_threshold.data.fill_(thr)
        ids = torch.randint(0, 64, (2, 12))
        out = m(input_ids=ids, labels=ids)
        assert torch.isfinite(out.loss).all(), f"[{label}] non-finite loss"
        a = m.audit()
        assert a["holds"], f"[{label}] σ certificate fails: {a}"
        # substrate readout stays within the port's bound in every mode
        r = m.substrate._read
        assert r.abs().max() <= m.substrate.output_bound + 1e-5, f"[{label}] readout unbounded"


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
