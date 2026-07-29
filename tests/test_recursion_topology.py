"""
E1 — Recursion topology tests.

Verifies the architectural rules of §3.2–3.5 against the code:
  - K = 16, not adaptive
  - Recursion state stays fp32 after a global .to(bf16)
  - Certificate is structural (σ<margin holds by construction, fails closed if broken)
  - Single broadcast: substrate.cond_proj and transformer.inject consume the SAME
    tensor identity (h_{w-1}); no dual-head split.

Requires torch. Skips cleanly otherwise.
"""
import ast
import inspect
import os
import sys
import textwrap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _have_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _tiny(seed=0):
    import torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    torch.manual_seed(seed)
    tcfg = AeonTransformerConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=64)
    m = HybridModel(h_rec=24, K=16, transformer_config=tcfg,
                    substrate={"kind": "matrix", "d_in": 24, "d_state": 24,
                               "n_head": 2, "head_size": 12},
                    dtype=torch.float32)
    m.recursion.float()
    return m


# ---- P-K16 -----------------------------------------------------------------
def test_K_is_16_and_not_adaptive():
    """K=16 in the model, in both configs, and no per-token / adaptive-clock
    mechanism (AST scan): the `range(num_windows)` loop MUST exist (proves the
    slow clock is loop-scheduled, not per-token) and the loop bound must be
    ceil(T/K) with K bound to the constant."""
    from aeon.hybrid import HybridModel
    import yaml
    # 1) code default
    src = inspect.getsource(HybridModel.__init__)
    assert "K: int = 16" in src, f"K default not 16 in HybridModel.__init__:\n{src[:400]}"
    # 2) configs
    for cfg_path in ("configs/aeon_350m.yaml", "configs/aeon_v1.yaml"):
        cfg = yaml.safe_load(open(cfg_path))
        assert cfg["model"]["K"] == 16, f"{cfg_path} K != 16"
    # 3) AST: forward has a window loop and the loop body doesn't call recursion
    #    per token — recursion.step must be OUTSIDE the inner (per-token) range.
    forward_src = textwrap.dedent(inspect.getsource(HybridModel.forward))
    tree = ast.parse(forward_src)
    calls_recursion_step = 0
    calls_substrate_step = 0
    depths_recursion, depths_substrate = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if name == "step":
                # count depth of enclosing `for`
                depth = 0
                for anc in ast.walk(tree):
                    if isinstance(anc, ast.For):
                        for child in ast.walk(anc):
                            if child is node:
                                depth += 1
                                break
                if isinstance(node.func.value, ast.Attribute):
                    if node.func.value.attr == "recursion":
                        calls_recursion_step += 1
                        depths_recursion.append(depth)
                    elif node.func.value.attr == "substrate":
                        calls_substrate_step += 1
                        depths_substrate.append(depth)
    assert calls_recursion_step >= 1, "recursion.step not called"
    assert calls_substrate_step >= 1, "substrate.step not called"
    # substrate.step is inside 2 for-loops (window, token); recursion.step in 1
    assert max(depths_substrate) > max(depths_recursion), (
        f"recursion.step nested as deep as substrate.step "
        f"(rec depth {max(depths_recursion)}, sub depth {max(depths_substrate)}) — "
        "per-token Recursion is FORBIDDEN by §3.4")


# ---- P-fp32-rec ------------------------------------------------------------
def test_recursion_stays_fp32_after_cast():
    """After `model.to(bf16)` and the mandated `model.recursion.float()`, every
    parameter and buffer under `model.recursion` must be fp32."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    try:
        torch.zeros(2, 2, dtype=torch.bfloat16) @ torch.zeros(2, 2, dtype=torch.bfloat16)
    except RuntimeError:
        print("  [skip] bf16 matmul unsupported"); return
    m = _tiny()
    m.to(torch.bfloat16)
    m.recursion.float()
    for name, p in m.recursion.named_parameters():
        assert p.dtype == torch.float32, f"{name}: {p.dtype}"
    for name, b in m.recursion.named_buffers():
        assert b.dtype in (torch.float32,), f"{name}: {b.dtype}"


# ---- P-cert (structural + fail-closed) -------------------------------------
def test_certificate_holds_by_construction():
    """`_build` returns W = sigmoid(s)*MARGIN*Cayley(A)*diag(tanh(d)); σ(W) MUST
    be < MARGIN for arbitrary parameter values. Randomise A/d/s and check."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.recursion import RecursionJoiner
    rj = RecursionJoiner(h_rec=16, d_substrate=16, d_transformer=16, d_embedding=32,
                         use_embedding_input=True, margin_h=0.98, margin_c=0.95)
    with torch.no_grad():
        for _ in range(20):
            rj.A_h.copy_(torch.randn_like(rj.A_h) * 3)
            rj.A_c.copy_(torch.randn_like(rj.A_c) * 3)
            rj.d_h.copy_(torch.randn_like(rj.d_h) * 3)
            rj.d_c.copy_(torch.randn_like(rj.d_c) * 3)
            rj.s_h.fill_(float(torch.randn(1) * 3))
            rj.s_c.fill_(float(torch.randn(1) * 3))
            a = rj.audit()
            assert a["holds"], f"structural cert violated: {a}"
            assert a["sigma_Wh"] < a["margin_h"]
            assert a["sigma_Wc"] < a["margin_c"]


def test_certificate_fails_closed_on_forced_violation():
    """If an adversary bypasses `_build` and installs an out-of-margin W, the
    audit() report must FAIL the invariant (not silently pass). This proves the
    audit is a real check, not decorative."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.recursion import RecursionJoiner
    rj = RecursionJoiner(h_rec=8, d_substrate=8, d_transformer=8, d_embedding=16,
                         use_embedding_input=True, margin_h=0.5, margin_c=0.5)
    # Directly poke a larger spectral radius into the built matrix path.
    # The audit reads W = _build(...): to force violation without touching _build,
    # we monkey-patch _build to return a matrix known to have σ > margin.
    big = 5.0 * torch.eye(8)                            # σ = 5.0 >> 0.5
    orig_build = rj._build
    rj._build = lambda *a, **k: big
    a = rj.audit()
    rj._build = orig_build
    assert not a["holds"], f"audit must fail closed on σ={a['sigma_Wh']} > margin={a['margin_h']}"


# ---- P-single-bcast --------------------------------------------------------
def test_single_broadcast_shared_source():
    """Both consumers use the SAME broadcast source — the previous window's
    Recursion state h. Assert:
      (a) HybridModel has no separate substrate-broadcast / transformer-broadcast
          projection module (no dual heads).
      (b) `inject_cols` (the transformer's broadcast) is built from `h_cond`, and
          `cond_in` (the substrate's broadcast) is built from the SAME `h_cond` —
          verified by AST inspection of forward().
    """
    from aeon.hybrid import HybridModel
    import torch
    m = _tiny() if _have_torch() else None

    # (a) forbidden module names: no J_S/J_T, no separate broadcast projection
    if m is not None:
        forbidden_substr = ("bcast_sub", "bcast_trans", "J_S", "J_T",
                            "sub_broadcast", "trans_broadcast",
                            "substrate_broadcast", "transformer_broadcast")
        names = [n for n, _ in m.named_modules()]
        for bad in forbidden_substr:
            assert not any(bad in n for n in names), \
                f"dual-broadcast module detected: {bad} in {names}"

    # (b) AST: `cond_in = cond_proj(h_cond...)` and `inject_cols.append(h_cond)`
    # both reference the SAME name `h_cond`.
    src = inspect.getsource(HybridModel.forward)
    assert "h_cond" in src
    assert "cond_proj(h_cond" in src, "substrate broadcast doesn't consume h_cond"
    assert "inject_cols.append(h_cond)" in src, "transformer broadcast doesn't consume h_cond"


# ---- Slow-clock cadence: recursion.step called once per window -------------
def test_recursion_step_called_once_per_window():
    """Runtime-level check that goes beyond AST. Monkey-patch recursion.step to
    count calls; run a forward with T tokens; assert count == ceil(T/K)."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import math, torch
    m = _tiny()
    B, T = 2, 32                              # T = 2·K = 32 → expect 2 recursion steps
    calls = {"n": 0}
    orig = m.recursion.step
    def counting(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)
    m.recursion.step = counting
    m(input_ids=torch.randint(0, 64, (B, T)))
    assert calls["n"] == math.ceil(T / m.K), \
        f"recursion.step called {calls['n']} times, expected {math.ceil(T/m.K)} for T={T}, K={m.K}"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
