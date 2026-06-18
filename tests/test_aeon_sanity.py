"""
Aeon-native sanity tests — no external reference, nothing external is
authoritative. Verifies the model's own contracts:

  1. forward pass produces correctly shaped logits (and loss with labels)
  2. the σ<margin certificate holds at init (Recursion.audit)
  3. gradient flows through every component (transformer, substrate, recursion,
     projections, γ)
  4. deterministic seed reproducibility (same seed -> identical logits)

Requires torch; skips cleanly otherwise. Uses a tiny fp32 config so it runs fast
on CPU. Run:  python tests/test_aeon_sanity.py   (or pytest)
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


def _tiny_model(seed=0):
    import torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    torch.manual_seed(seed)
    tcfg = AeonTransformerConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=128, tie_word_embeddings=True,
    )
    model = HybridModel(
        h_rec=32, K=4, transformer_config=tcfg,
        substrate={"kind": "matrix", "d_in": 32, "d_state": 32, "n_head": 2, "head_size": 16},
        freeze_backbone=False, use_embedding_input=True, dtype=torch.float32,
    )
    model.recursion.float()
    return model, tcfg


def test_forward_shapes():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    model, tcfg = _tiny_model()
    B, T = 2, 8
    ids = torch.randint(0, tcfg.vocab_size, (B, T))
    out = model(input_ids=ids, labels=ids)
    assert out.logits.shape == (B, T, tcfg.vocab_size), out.logits.shape
    assert out.loss is not None and out.loss.ndim == 0
    assert torch.isfinite(out.loss).all()


def test_certificate_holds_at_init():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    model, _ = _tiny_model()
    a = model.audit()
    assert a["holds"], f"σ certificate fails at init: {a}"
    assert a["sigma_Wh"] < a["margin_h"] and a["sigma_Wc"] < a["margin_c"]


def test_gradient_flows_everywhere():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    model, tcfg = _tiny_model()
    ids = torch.randint(0, tcfg.vocab_size, (2, 8))
    model(input_ids=ids, labels=ids).loss.backward()
    # representative parameter from each component must receive gradient
    checks = {
        "transformer.embed": model.transformer.model.embed_tokens.weight,
        "transformer.attn_q": model.transformer.model.layers[0].self_attn.q_proj.weight,
        "transformer.read_proj": model.transformer.read_proj.weight,
        "transformer.write_proj": model.transformer.write_proj.weight,
        "transformer.gamma": model.transformer.gamma,
        "substrate.key": model.substrate.key.weight,
        "recursion.W_s": model.recursion.W_s.weight,
        "recursion.A_h": model.recursion.A_h,
        "hybrid.emb_proj": model.emb_proj.weight,
        "hybrid.s_proj": model.s_proj.weight,
    }
    missing = [name for name, p in checks.items()
               if p.grad is None or p.grad.abs().sum().item() == 0.0]
    assert not missing, f"no gradient into: {missing}"


def test_seed_reproducibility():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    m1, tcfg = _tiny_model(seed=7)
    m2, _ = _tiny_model(seed=7)
    torch.manual_seed(123); ids = torch.randint(0, tcfg.vocab_size, (2, 8))
    with torch.no_grad():
        l1 = m1(input_ids=ids).logits
        l2 = m2(input_ids=ids).logits
    assert torch.equal(l1, l2), f"same seed diverged: max|Δ|={(l1-l2).abs().max().item()}"


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
