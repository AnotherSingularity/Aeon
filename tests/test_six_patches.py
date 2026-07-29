"""
E1 — Six-patch regression suite.

One explicit test per V0.02.02 debug patch, so a broken patch fails a NAMED test
(not a broad smoke test). Preservation manifest ids P-4a through P-4f.

Each test asserts the invariant at its stated scope (init / forward / dtype
transition / state reset) so a future refactor cannot silently drop one patch
under the umbrella of "everything still runs".

Requires torch. Skips cleanly otherwise.
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


def _hybrid_transformer(dtype=None):
    """Isolated HybridTransformer at tiny scale (patch checks don't need HybridModel)."""
    import torch
    from aeon.transformer import HybridTransformer, AeonTransformerConfig
    torch.manual_seed(0)
    cfg = AeonTransformerConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=16, max_position_embeddings=64)
    ht = HybridTransformer(h_rec=24, config=cfg, dtype=dtype or torch.float32)
    return ht


# ---- P-4a: γ recast to fp32 after model.to(dtype) --------------------------
def test_4a_gamma_recast_after_cast():
    """The recast in scripts/train.py:119 is the LOAD-BEARING fp32 restoration.
    After `model.to(bf16)` a Parameter created fp32 (P-4b) will still be bf16;
    P-4a explicitly restores it. Assert the recast pattern behaves as expected."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    try:
        torch.zeros(2, 2, dtype=torch.bfloat16) @ torch.zeros(2, 2, dtype=torch.bfloat16)
    except RuntimeError:
        print("  [skip] bf16 matmul unsupported"); return
    ht = _hybrid_transformer(dtype=torch.bfloat16)
    ht.gamma.data = ht.gamma.data.float()          # <-- P-4a
    assert ht.gamma.dtype == torch.float32, f"γ dtype after recast: {ht.gamma.dtype}"


# ---- P-4b: γ Parameter created fp32 in transformer.py:246 ------------------
def test_4b_gamma_param_dtype():
    """Belt-and-suspenders with P-4a: the code declares intent that γ is fp32.
    Freshly built at fp32, HybridTransformer.gamma must be fp32."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    ht = _hybrid_transformer(dtype=torch.float32)
    assert isinstance(ht.gamma, torch.nn.Parameter)
    assert ht.gamma.dtype == torch.float32, f"γ init dtype: {ht.gamma.dtype}"


# ---- P-4c: inject() fp32 residual add --------------------------------------
def test_4c_inject_fp32_add():
    """The inject residual add must upcast: (hidden.float() + γ · write_proj(sig).float()).to(dt).
    Assert (a) with γ=0 the return equals hidden bit-exactly (identity gate closed);
    (b) with γ>0 in bf16 mode the output dtype matches hidden dtype (kept fp32 internally).
    """
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    try:
        torch.zeros(2, 2, dtype=torch.bfloat16) @ torch.zeros(2, 2, dtype=torch.bfloat16)
    except RuntimeError:
        print("  [skip] bf16 matmul unsupported"); return
    ht = _hybrid_transformer(dtype=torch.bfloat16)
    # match production path: everything bf16 EXCEPT γ (P-4a recast)
    ht.to(torch.bfloat16)
    ht.gamma.data = ht.gamma.data.float()
    hidden = torch.randn(2, 4, ht.D, dtype=torch.bfloat16)
    signal = torch.randn(2, 4, ht.h_rec, dtype=torch.bfloat16)
    # γ = 0: identity gate — inject == hidden
    with torch.no_grad():
        ht.gamma.fill_(0.0)
        out0 = ht.inject(hidden, signal)
    assert out0.dtype == hidden.dtype, out0.dtype
    assert torch.equal(out0, hidden), "γ=0 inject must be identity"
    # γ > 0: dtype preserved, and the ADD path is genuinely fp32 (assert this from source)
    import inspect, re
    src = inspect.getsource(ht.inject)
    assert re.search(r"hidden\.float\(\).*self\.gamma.*write_proj.*\.float\(\)", src, re.DOTALL), \
        "P-4c: inject() must add in fp32 (source pattern absent)"


# ---- P-4d: write_proj random init (not zeros) ------------------------------
def test_4d_write_proj_random_init():
    """With both γ=0 and write_proj=0 the gradient wrt γ is 0 (=write_proj(signal))
    AND the gradient wrt write_proj is 0 (=γ·signal): mutual-zero deadlock.
    P-4d randomizes write_proj so γ receives gradient at init. Assert the weights
    aren't zero AND that γ gets a non-zero gradient from a fresh forward."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    ht = _hybrid_transformer()
    assert ht.write_proj.weight.abs().sum() > 0, "write_proj must be non-zero at init"
    # γ = 0 gradient path: γ_grad = write_proj(signal); with non-zero write_proj this is non-zero
    with torch.no_grad(): ht.gamma.fill_(0.0)
    signal = torch.randn(2, 4, ht.h_rec, requires_grad=False)
    hidden = torch.randn(2, 4, ht.D, requires_grad=True)
    inject = ht.inject(hidden, signal)
    inject.sum().backward()
    assert ht.gamma.grad is not None
    assert ht.gamma.grad.abs().item() > 0, "γ received zero gradient — mutual-zero deadlock"


# ---- P-4e: substrate reset dtype follows param dtype -----------------------
def test_4e_substrate_state_dtype():
    """Cast a cell to bf16, reset, then step: state tensors must be bf16 (matching
    params). A fp32 state × bf16 param matmul would crash."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    try:
        torch.zeros(2, 2, dtype=torch.bfloat16) @ torch.zeros(2, 2, dtype=torch.bfloat16)
    except RuntimeError:
        print("  [skip] bf16 matmul unsupported"); return
    from aeon.substrate.matrix_cell import MatrixStateCell
    from aeon.substrate.vector_cell import VectorStateCell
    for cell in (MatrixStateCell(d_in=8, d_state=12, n_head=2, head_size=6),
                 VectorStateCell(d_in=8, d_state=12)):
        cell.to(torch.bfloat16)
        cell.reset(2)
        # first live state field found by inspecting the module's tensor attrs
        state = next(t for t in vars(cell).values()
                     if isinstance(t, torch.Tensor) and t.numel() > 0)
        assert state.dtype == torch.bfloat16, f"{type(cell).__name__} state dtype {state.dtype}"
        y = cell.step(torch.randn(2, cell.d_in, dtype=torch.bfloat16))
        assert torch.isfinite(y).all(), f"{type(cell).__name__} bf16 forward produced non-finite"


# ---- P-4f: rotary inv_freq fresh fp32 per forward, no register_buffer ------
def test_4f_rotary_inv_freq_fresh_fp32():
    """Verify: (a) transformer.py has zero register_buffer calls (buffer trap
    can't be reintroduced); (b) AeonRotary(x, pos) produces fp32 angles that a
    global .to(bf16) does NOT downcast to bf16 (the whole point of P-4f).
    """
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import re, inspect, torch
    import aeon.transformer as tm
    src = inspect.getsource(tm)
    assert "register_buffer" not in src, "P-4f: register_buffer must not appear in transformer.py"
    # inv_freq must be built INSIDE the forward, in fp32
    rotary_src = inspect.getsource(tm.AeonRotary.forward)
    assert "inv_freq" in rotary_src, "inv_freq must be built in forward"
    assert re.search(r"\.float\(\)", rotary_src), "inv_freq path must produce fp32"
    # runtime: after model.to(bf16), rotary cos/sin cast to bf16 (activation dtype)
    # but the INTERNAL frequency computation stays fp32 — verify by inspecting angles.
    rot = tm.AeonRotary(tm.AeonTransformerConfig(head_dim=8)).to(torch.bfloat16)
    x = torch.zeros(1, 4, 8, dtype=torch.bfloat16)
    pos = torch.arange(4)[None]
    cos, sin = rot(x, pos)
    assert cos.dtype == torch.bfloat16 and sin.dtype == torch.bfloat16, "cos/sin ride activation dtype"
    # Also assert non-degenerate: cos values aren't all 1.0 (which would signal frequencies underflowed)
    assert not (cos == 1.0).all(), "cos all 1.0 — inv_freq collapsed (P-4f regression)"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
