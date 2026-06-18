"""
Conformance tests for the substrate port (§d).

These are thin shells over `aeon.substrate.verify_substrate` — the single
conformance entry point. The contract logic lives in the utility, not here:
  * torch-free: the utility's static checks (capability negotiation, decay
    read-only, contract attributes) run against pure-Python mocks, and the AST
    anti-drift mechanism runs against the real vector_cell source;
  * torch: the full suite (incl. execution + per-cell structural checks) runs
    against the real cells, skipping cleanly when torch is unavailable.

Runnable directly (`python3 tests/test_substrate_port.py`) and under pytest.
"""
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from aeon.substrate import (  # noqa: E402
    SubstratePort,
    make_substrate,
    verify_substrate,
    make_ast_drift_check,
    MATRIX_READ,
)


# --- pure-Python substrates exercising the utility's torch-free checks ------
def _clamp1(x):
    return max(-1.0, min(1.0, x))


class MockSubstrate(SubstratePort):
    CAPABILITIES = frozenset()

    def __init__(self, d_in=3, d_state=2):
        self.d_in, self.d_state, self.output_bound = d_in, d_state, 1.0
        self._read = None

    def reset(self, batch_size, device=None):
        self._read = [[0.0] * self.d_state for _ in range(batch_size)]

    def step(self, x_t):
        self._read = [[_clamp1(sum(row))] * self.d_state for row in x_t]
        return self._read

    def read(self):
        return self._read

    def write(self, drive):
        pass


class MockRichSubstrate(MockSubstrate):
    CAPABILITIES = frozenset({MATRIX_READ})

    def read_matrix(self):
        return "S"


# --- torch-free: utility static checks via mocks ----------------------------
def test_verify_mock_static():
    report = verify_substrate(MockSubstrate())
    assert report.ok, str(report)
    statuses = {r.name: r.status for r in report.results}
    # static port checks pass; execution checks skip (mock is not a torch module)
    assert statuses["capabilities_subset"] == "pass"
    assert statuses["decay_read_only"] == "pass"
    assert statuses["unadvertised_ports_gated"] == "pass"
    assert statuses["required_tier_shapes"] == "skip"


def test_verify_rich_mock_static():
    report = verify_substrate(MockRichSubstrate())
    assert report.ok, str(report)
    statuses = {r.name: r.status for r in report.results}
    assert statuses["advertised_ports_implemented"] == "pass"


# --- torch-free: AST anti-drift mechanism against the real vector_cell source -
def test_vector_cell_source_anti_drift():
    chk = make_ast_drift_check(
        "aeon.substrate.vector_cell",
        forbidden={"spectral_norm", "carry", "ema", "gate",
                   "forget", "clamp", "clip", "sigmoid"},
        required={"tanh", "W_x", "W_h", "scalar"},
    )
    chk()  # raises on drift; torch-free (locates source without importing it)


def test_factory_unknown_kind_raises():
    for bad in ({}, {"kind": "nope", "d_in": 1, "d_state": 1}):
        try:
            make_substrate(bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError from make_substrate")


# --- torch: full suite against the real cells -------------------------------
def _have_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def test_verify_real_cells():
    if not _have_torch():
        print("  [skip] torch unavailable — real-cell conformance skipped")
        return
    for cfg in ({"kind": "matrix", "d_in": 8, "d_state": 4},
                {"kind": "vector", "d_in": 8, "d_state": 6}):
        report = verify_substrate(make_substrate(cfg))
        assert report.ok, "\n" + str(report)


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")
    # show a sample report for visibility
    print("\nSample report (MockSubstrate):")
    print(verify_substrate(MockSubstrate()))


if __name__ == "__main__":
    _run_all()
