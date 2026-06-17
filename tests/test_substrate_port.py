"""
Conformance tests for the substrate port (§d).

Layers:
  1. Contract tests — framework-free. Capability negotiation, required/optional
     gating, and the bounded-output contract, via a pure-Python mock. Run
     anywhere (no torch).
  2. Structural anti-drift test — framework-free. Parses vru_cell.py and asserts
     it contains no Recursion-class mechanisms (spectral norm, carry/EMA, gates,
     clamping). Durable guard against the conflation recurring.
  3. Cell tests — verify the concrete RWKV / VRU cells satisfy the required tier,
     advertise the expected capabilities, keep decay read-only, return bounded
     output, and (VRU) have the disclosed single-state structure. Skipped when
     torch is unavailable.

Runnable directly (`python3 tests/test_substrate_port.py`) and under pytest.
"""
import ast
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from aeon.substrate import (  # noqa: E402
    SubstratePort,
    CapabilityError,
    make_substrate,
    MATRIX_READ,
    DECAY_CONTROL,
    ASSOC_WRITE,
)


# --- a pure-Python substrate that satisfies only the required tier ----------
def _clamp1(x):
    return max(-1.0, min(1.0, x))


class MockSubstrate(SubstratePort):
    CAPABILITIES = frozenset()

    def __init__(self, d_in=3, d_state=2):
        self.d_in = d_in
        self.d_state = d_state
        self.output_bound = 1.0
        self._read = None
        self._drive = None

    def reset(self, batch_size, device=None):
        self._read = [[0.0] * self.d_state for _ in range(batch_size)]
        self._drive = None

    def step(self, x_t):
        # readout is the (bounded) row-sum of the input — respects output_bound
        self._read = [[_clamp1(sum(row))] * self.d_state for row in x_t]
        self._drive = None
        return self._read

    def read(self):
        return self._read

    def write(self, drive):
        self._drive = drive


class MockRichSubstrate(MockSubstrate):
    CAPABILITIES = frozenset({MATRIX_READ})

    def read_matrix(self):
        return "S"


# --- contract tests (no torch) ----------------------------------------------
def test_required_tier_present():
    s = MockSubstrate()
    s.reset(2)
    out = s.step([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
    assert out == s.read()
    assert len(out) == 2 and len(out[0]) == s.d_state
    s.write("drive")


def test_capability_negotiation():
    plain, rich = MockSubstrate(), MockRichSubstrate()
    assert plain.capabilities() == frozenset()
    assert not plain.has(MATRIX_READ)
    assert rich.has(MATRIX_READ)
    assert rich.read_matrix() == "S"


def test_optional_gating_raises():
    plain = MockSubstrate()
    for call in (plain.read_matrix, plain.read_decay,
                 lambda: plain.assoc_write(0, 0)):
        try:
            call()
        except CapabilityError:
            continue
        raise AssertionError("expected CapabilityError on unadvertised capability")


def test_bounded_output_contract_mock():
    s = MockSubstrate()
    s.reset(2)
    out = s.step([[100.0, 100.0, 100.0], [-50.0, -50.0, 0.0]])
    for row in out:
        for val in row:
            assert abs(val) <= s.output_bound + 1e-6


def test_factory_unknown_kind_raises():
    for bad in ({}, {"kind": "nope", "d_in": 1, "d_state": 1}):
        try:
            make_substrate(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError from make_substrate")


# --- structural anti-drift test (no torch) ----------------------------------
def test_vru_no_recursion_class_mechanisms():
    """STRUCTURAL guard (Issue 1). The candidate substrate must not drift back
    into Recursion-class. Parse the source and assert no forbidden mechanisms
    appear as identifiers, and that the disclosed recurrence form is present.
    Source-level so comments/docstrings (which legitimately *name* the avoided
    mechanisms) do not trip it."""
    src = open(os.path.join(REPO_ROOT, "aeon", "substrate", "vru_cell.py")).read()
    tree = ast.parse(src)

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, ast.alias):
            names.add(node.name)
            if node.asname:
                names.add(node.asname)
    lowered = {n.lower() for n in names}

    forbidden = {
        "spectral_norm",   # Recursion's σ<1 bound, not the candidate's
        "carry",           # no carry stream
        "ema",             # no EMA carry
        "gate", "forget",  # no gates
        "clamp", "clip",   # fixed scalar, no clamping
        "sigmoid",         # no gated/decay-logit mechanism
    }
    hit = forbidden & lowered
    assert not hit, f"VRU drifted toward Recursion-class: forbidden {sorted(hit)}"
    assert "_c" not in names, "VRU has a second (carry) state attribute"

    # disclosed recurrence form present: tanh(W_x @ x + scalar * W_h @ h)
    for required in ("tanh", "W_x", "W_h", "scalar"):
        assert required in names, f"VRU missing recurrence element {required!r}"


# --- cell tests (require torch) ---------------------------------------------
def _have_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def test_cells_required_tier_and_capabilities():
    if not _have_torch():
        print("  [skip] torch unavailable — cell numeric tests skipped")
        return
    import torch

    B, d_in, d_state = 2, 8, 4
    cases = (
        ({"kind": "rwkv", "d_in": d_in, "d_state": d_state},
         {MATRIX_READ, DECAY_CONTROL, ASSOC_WRITE}),
        ({"kind": "vru", "d_in": d_in, "d_state": d_state},
         {DECAY_CONTROL}),
    )
    for cfg, expected in cases:
        cell = make_substrate(cfg)
        assert isinstance(cell, SubstratePort)
        assert cell.capabilities() == frozenset(expected)
        cell.reset(B)
        out = cell.step(torch.randn(B, d_in))
        assert out.shape == (B, d_state)
        assert cell.read().shape == (B, d_state)
        cell.write(torch.randn(B, d_state))
        assert cell.step(torch.randn(B, d_in)).shape == (B, d_state)
        if cell.has(MATRIX_READ):
            assert cell.read_matrix().shape[0] == B and cell.read_matrix().dim() == 4


def test_decay_is_read_only():
    """Issue 2: decay is substrate-owned; the port exposes read-only
    introspection and no mutator."""
    if not _have_torch():
        print("  [skip] torch unavailable — decay read-only test skipped")
        return
    # no mutator anywhere in the port surface
    assert not hasattr(SubstratePort, "set_decay")
    for cfg in ({"kind": "rwkv", "d_in": 8, "d_state": 4},
                {"kind": "vru", "d_in": 8, "d_state": 4}):
        cell = make_substrate(cfg)
        assert cell.has(DECAY_CONTROL)
        assert not hasattr(cell, "set_decay")
        cell.read_decay()  # callable, returns a descriptor (scalar or tensor)


def test_bounded_output_cells():
    """Issue 3: every cell's step output is finite and within output_bound."""
    if not _have_torch():
        print("  [skip] torch unavailable — cell boundedness test skipped")
        return
    import torch

    B, d_in, d_state = 3, 8, 4
    for cfg in ({"kind": "rwkv", "d_in": d_in, "d_state": d_state},
                {"kind": "vru", "d_in": d_in, "d_state": d_state}):
        cell = make_substrate(cfg)
        cell.reset(B)
        for _ in range(8):
            out = cell.step(torch.randn(B, d_in) * 50.0)  # large drive
            assert torch.isfinite(out).all()
            assert out.abs().max().item() <= cell.output_bound + 1e-5


def test_vru_disclosed_structure():
    """Issue 1 (runtime half): VRU has exactly one state tensor of dim H, no
    forbidden parameter names, a fixed-float scalar, and gradient flows through
    both W_x and W_h (recurrence form), verified structurally — not by numeric
    comparison to reference outputs."""
    if not _have_torch():
        print("  [skip] torch unavailable — VRU structure test skipped")
        return
    import torch

    B, d_in, H = 2, 8, 6
    cell = make_substrate({"kind": "vru", "d_in": d_in, "d_state": H})

    # fixed geometric scalar, not a Parameter
    assert isinstance(cell.scalar, float)
    pnames = dict(cell.named_parameters())
    assert "scalar" not in pnames

    # no forbidden parameter names
    for name in pnames:
        low = name.lower()
        assert not any(f in low for f in ("gate", "carry", "forget", "input", "output")), name

    # exactly one runtime state tensor, of shape (B, H)
    cell.reset(B)
    cell.step(torch.randn(B, d_in))
    state_tensors = {k: v for k, v in vars(cell).items()
                     if isinstance(v, torch.Tensor)}
    assert len(state_tensors) == 1, f"expected one state tensor, got {list(state_tensors)}"
    (h,) = state_tensors.values()
    assert h.shape == (B, H)

    # gradient flows through both W_x and W_h (recurrence uses both)
    cell.reset(B)
    cell.step(torch.randn(B, d_in))            # h now non-zero
    loss = cell.step(torch.randn(B, d_in)).sum()
    loss.backward()
    assert cell.W_x.weight.grad is not None
    assert cell.W_h.weight.grad is not None and cell.W_h.weight.grad.abs().sum() > 0


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
