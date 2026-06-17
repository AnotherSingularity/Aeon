"""
Conformance tests for the substrate port (§d).

Two layers:
  1. Contract tests — framework-free. They verify capability negotiation and
     required/optional gating using a pure-Python mock substrate. These run
     anywhere (no torch needed).
  2. Cell tests — verify the concrete RWKV / VRU cells satisfy the required tier
     and advertise the expected optional capabilities. Skipped when torch is
     unavailable.

Runnable directly (`python3 tests/test_substrate_port.py`) and under pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aeon.substrate import (  # noqa: E402
    SubstratePort,
    CapabilityError,
    make_substrate,
    MATRIX_READ,
    DECAY_CONTROL,
    ASSOC_WRITE,
)


# --- a pure-Python substrate that satisfies only the required tier ----------
class MockSubstrate(SubstratePort):
    CAPABILITIES = frozenset()

    def __init__(self, d_in=3, d_state=2):
        self.d_in = d_in
        self.d_state = d_state
        self._read = None
        self._drive = None

    def reset(self, batch_size, device=None):
        self._read = [[0.0] * self.d_state for _ in range(batch_size)]
        self._drive = None

    def step(self, x_t):
        # trivial: readout is the row-sum of the input, plus any staged drive
        self._read = [[sum(row)] * self.d_state for row in x_t]
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
    s.write("drive")  # required write port accepts a drive


def test_capability_negotiation():
    plain, rich = MockSubstrate(), MockRichSubstrate()
    assert plain.capabilities() == frozenset()
    assert not plain.has(MATRIX_READ)
    assert rich.has(MATRIX_READ)
    assert rich.read_matrix() == "S"


def test_optional_gating_raises():
    plain = MockSubstrate()
    for call in (plain.read_matrix, lambda: plain.set_decay(0),
                 lambda: plain.assoc_write(0, 0)):
        try:
            call()
        except CapabilityError:
            continue
        raise AssertionError("expected CapabilityError on unadvertised capability")


def test_factory_unknown_kind_raises():
    for bad in ({}, {"kind": "nope", "d_in": 1, "d_state": 1}):
        try:
            make_substrate(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError from make_substrate")


# --- cell tests (require torch) ---------------------------------------------
def _torch_or_skip():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def test_cells_required_tier_and_capabilities():
    if not _torch_or_skip():
        print("  [skip] torch unavailable — cell numeric tests skipped")
        return
    import torch

    B, d_in, d_state = 2, 8, 4
    for cfg, expected in (
        ({"kind": "rwkv", "d_in": d_in, "d_state": d_state}, {MATRIX_READ, DECAY_CONTROL, ASSOC_WRITE}),
        ({"kind": "vru", "d_in": d_in, "d_state": d_state}, {DECAY_CONTROL}),
    ):
        cell = make_substrate(cfg)
        assert isinstance(cell, SubstratePort)
        assert cell.capabilities() == frozenset(expected)
        cell.reset(B)
        x = torch.randn(B, d_in)
        out = cell.step(x)
        assert out.shape == (B, d_state)
        assert cell.read().shape == (B, d_state)
        cell.write(torch.randn(B, d_state))
        out2 = cell.step(torch.randn(B, d_in))
        assert out2.shape == (B, d_state)
        if cell.has(MATRIX_READ):
            S = cell.read_matrix()
            assert S.shape[0] == B and S.dim() == 4


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
