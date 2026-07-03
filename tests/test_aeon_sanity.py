"""
Aeon-native sanity tests — no external reference, nothing external is
authoritative. Verifies the model's own contracts:

  1. forward pass produces correctly shaped logits (and loss with labels)
  2. the σ<margin certificate holds at init (Recursion.audit)
  3. gradient flows through every component (transformer, substrate, recursion,
     projections, γ)
  4. deterministic seed reproducibility (same seed -> identical logits)
  5. γ actually updates under the optimizer (the bf16-trap regression: fp32 master
     γ moves by more than the bf16 ULP it would otherwise freeze to)
  6. no external model/architecture library participates in the forward path

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
    """Every trainable component receives gradient when the write gate is ENGAGED.

    Two design facts shape this test:
      * The recurrent branch (substrate / Recursion / read/write/s/emb projections)
        reaches the loss ONLY through the γ-gated inject(). At the γ=0 warm-start,
        that gate is closed by construction, so those params get zero gradient
        while γ itself still does (it is what lifts the gate). We therefore engage
        the gate (γ>0) to test the trained regime.
      * recursion.A_h drives W_h, the recurrent-state matrix. It only earns
        gradient when a NON-zero carried state reaches the loss, which needs at
        least three K-windows (window 2's state is computed from window 1's
        non-zero carry). We use T = 3·K so that carry path is exercised.
    """
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    model, tcfg = _tiny_model()
    with torch.no_grad():
        model.transformer.gamma.fill_(0.05)          # engage the write gate
    T = 3 * model.K                                   # ≥3 windows so W_h/A_h is exercised
    ids = torch.randint(0, tcfg.vocab_size, (2, T))
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


def test_gamma_updates_bf16_trap():
    """The load-bearing regression: γ must actually move under the optimizer.

    A bf16 γ near 0.03 has ULP ≈ 2^-12 ≈ 2.44e-4 (exponent 2^-5), which is above
    AdamW's per-step update (~lr), so a bf16 γ freezes at 1/32 and never crosses.
    The fix is to keep γ an fp32 master parameter after the global dtype cast.
    This test recreates the trap conditions — cast the model to bf16, seed γ near
    0.03 — then applies the fix and verifies γ moves by MORE than the bf16 ULP it
    would otherwise be quantised to (i.e. movement a bf16 γ could not represent).
    """
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    try:
        torch.zeros(2, 2, dtype=torch.bfloat16) @ torch.zeros(2, 2, dtype=torch.bfloat16)
    except RuntimeError:
        print("  [skip] bf16 matmul unsupported on this device"); return

    model, tcfg = _tiny_model()
    model.to(dtype=torch.bfloat16)          # the trap: casts every param, γ included
    model.recursion.float()                 # certificate stays fp32 (as in training)
    # the fix under test — γ back to an fp32 master parameter after the cast:
    model.transformer.gamma.data = model.transformer.gamma.data.float()
    with torch.no_grad():
        model.transformer.gamma.fill_(0.03)  # seed near the freeze point
    assert model.transformer.gamma.dtype == torch.float32, "γ must be fp32 after the fix"

    gamma0 = model.transformer.gamma.item()
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-4)
    torch.manual_seed(0)
    max_dev = 0.0
    for _ in range(10):
        ids = torch.randint(0, tcfg.vocab_size, (2, 8))
        opt.zero_grad(set_to_none=True)
        model(input_ids=ids, labels=ids).loss.backward()
        assert model.transformer.gamma.grad is not None, "γ received no gradient"
        opt.step()
        max_dev = max(max_dev, abs(model.transformer.gamma.item() - gamma0))

    bf16_ulp_at_0p03 = 2.0 ** -12            # ≈ 2.44e-4, the freeze granularity
    assert max_dev > bf16_ulp_at_0p03, (
        f"γ moved only {max_dev:.2e} (≤ bf16 ULP {bf16_ulp_at_0p03:.2e}) — "
        f"the bf16 trap fix is not effective")


def test_no_external_library_in_forward():
    """No external model/architecture library may participate in the forward path.
    Import Aeon, run a forward+backward, then assert none of the banned families
    made it into sys.modules — nothing external is imported, let alone in the graph.
    """
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import sys
    import torch
    import aeon  # noqa: F401  — importing Aeon must pull in nothing external
    model, tcfg = _tiny_model()
    ids = torch.randint(0, tcfg.vocab_size, (2, 8))
    model(input_ids=ids, labels=ids).loss.backward()  # exercise the full graph

    # Guard on LIBRARY packages, not architecture names: any external
    # architecture could only reach the forward path THROUGH one of these
    # third-party libraries, and none of them may be imported. torch / numpy /
    # yaml / safetensors are Aeon's sanctioned generic dependencies and are fine.
    forbidden_libs = ("transformers", "accelerate", "datasets",
                      "deepspeed", "flash_attn", "xformers")
    present = sorted({m for m in sys.modules
                      for b in forbidden_libs if m == b or m.startswith(b + ".")})
    assert not present, f"external library present after Aeon forward: {present}"


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
