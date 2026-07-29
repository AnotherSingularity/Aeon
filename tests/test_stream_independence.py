"""
E1 — Stream-independence contract tests.

§3.1 says the two streams are independent. §3.6 says the substrate (including its
gate) may not read transformer state.

Two levels of guard:
  1. Static (AST): the substrate package (aeon.substrate.*) must NOT import from
     aeon.transformer, and vice versa; the substrate cell / feedback controller
     must not reference transformer-side names.
  2. Runtime: swap the transformer's forward output with a marker after
     substrate.step() runs, and prove the substrate produces the same readout as
     when the transformer's output was intact (the substrate's outputs are
     independent of the transformer's).

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


# ---- P-parallel: no direct cross-stream import ------------------------------
def test_substrate_does_not_import_transformer():
    """AST-scan the substrate package: no import touches aeon.transformer.
    (aeon.hybrid IS allowed to import both — it's the integration file.)"""
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "aeon", "substrate")
    for fname in sorted(os.listdir(root)):
        if not fname.endswith(".py"):
            continue
        path = os.path.join(root, fname)
        tree = ast.parse(open(path).read())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                assert "aeon.transformer" not in mod, \
                    f"{fname} imports from aeon.transformer: {mod}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert "aeon.transformer" not in alias.name, \
                        f"{fname} imports aeon.transformer: {alias.name}"


def test_transformer_does_not_import_substrate():
    """Mirror check for the transformer side."""
    tm_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "aeon", "transformer.py")
    tree = ast.parse(open(tm_path).read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert "aeon.substrate" not in mod, f"transformer.py imports {mod}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                assert "aeon.substrate" not in alias.name, \
                    f"transformer.py imports {alias.name}"


# ---- P-sub-autonomy: substrate feedback is autonomous -----------------------
def test_substrate_feedback_uses_no_transformer_names():
    """AST-scan the feedback controller: no identifier / attribute in code
    references a transformer-side surface. (Substring on the raw source would
    false-positive on doc comparisons like "std 0.02, like the write_proj patch";
    AST walks the code proper — Names, Attributes, function args, arg names.)"""
    from aeon.substrate import feedback as fb_mod
    src = textwrap.dedent(inspect.getsource(fb_mod))
    tree = ast.parse(src)
    forbidden = {"hidden_states", "logits", "attention", "attn", "entropy",
                 "read_proj", "write_proj", "AeonTransformer", "lm_head",
                 "embed_tokens", "transformer"}
    ident_hits = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden:
            ident_hits.add(node.id)
        elif isinstance(node, ast.Attribute) and node.attr in forbidden:
            ident_hits.add(node.attr)
        elif isinstance(node, ast.arg) and node.arg in forbidden:
            ident_hits.add(node.arg)
    assert not ident_hits, f"feedback.py references transformer-side identifiers: {ident_hits}"


def test_matrix_cell_step_signature_is_substrate_only():
    """The substrate cell's step() takes only x_t. No transformer signal is
    plumbed in as an argument — the substrate consumes only its input token and
    the (already-broadcast) authorized Recursion carry via write()."""
    from aeon.substrate.matrix_cell import MatrixStateCell
    from aeon.substrate.vector_cell import VectorStateCell
    for cls in (MatrixStateCell, VectorStateCell):
        sig = inspect.signature(cls.step)
        # (self, x_t) — no extra transformer-derived arg
        params = list(sig.parameters)
        assert params == ["self", "x_t"], f"{cls.__name__}.step signature: {params}"


# ---- Runtime: substrate step output is invariant to transformer state -------
def _tiny_model(seed=0):
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


def test_substrate_readout_invariant_to_transformer_within_window():
    """RUNTIME proof of §3.1 (no DIRECT read): perturb the transformer's hidden
    output and confirm the substrate's readouts WITHIN THE FIRST WINDOW are bit-
    identical. §3.2 allows the transformer to reach the substrate through the
    Recursion broadcast at window boundaries — so from token K onward divergence
    is EXPECTED (that IS the authorised integration path). This test asserts
    exactly the §3.1 boundary: no cross-stream leak inside a window."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    m = _tiny_model()
    K = m.K
    ids = torch.randint(0, 64, (2, 2 * K))       # two windows: check ONLY the first

    captured = {"normal": [], "perturbed": []}
    orig_step = m.substrate.step

    def capture_into(bucket):
        def wrapped(x):
            y = orig_step(x)
            bucket.append(y.detach().clone())
            return y
        return wrapped

    m.substrate.step = capture_into(captured["normal"])
    m(input_ids=ids)

    orig_hidden = m.transformer.hidden_states
    def perturbed_hidden(*a, **k):
        return orig_hidden(*a, **k) + 100.0      # massive transformer-side perturbation
    m.transformer.hidden_states = perturbed_hidden
    m.substrate.step = capture_into(captured["perturbed"])
    m(input_ids=ids)

    m.transformer.hidden_states = orig_hidden
    m.substrate.step = orig_step

    # Window 0 (tokens 0..K-1): substrate MUST be bit-identical — no direct read.
    for i in range(K):
        a, b = captured["normal"][i], captured["perturbed"][i]
        assert torch.equal(a, b), (
            f"§3.1 VIOLATION: window-0 substrate readout token {i} depends on "
            f"transformer state (max|Δ|={(a-b).abs().max().item()})")
    # Window 1: divergence is EXPECTED and PROVES the authorised path — the
    # transformer reaches the substrate via the Recursion broadcast at the K
    # boundary. Positive assertion: divergence must be non-zero.
    a1, b1 = captured["normal"][K], captured["perturbed"][K]
    assert not torch.equal(a1, b1), (
        "authorised Recursion-mediated cross-stream integration is silent — "
        "either the broadcast is not being consumed, or the perturbation is trivial")


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
