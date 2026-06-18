"""
aeon/substrate/vector_cell.py — Aeon vector-state recurrent cell.

A deliberately simple signal-source cell: a single state vector h of dimension
H, no gates, no carry stream, no certificate machinery. It is intentionally
distinct from Recursion (which owns the σ<1 contractive manifold) — this cell is
the lightweight, high-throughput alternative behind the same port.

Spec:
    state      : a single tensor h of dimension H (= d_state)
    recurrence : h_new = tanh(W_x @ x + scalar * W_h @ h)
    scalar     : a fixed geometric constant (not learned, not clamped)
    no gates, no carry stream, no split state

Output is h itself: h = tanh(...) ∈ (-1, 1)^H, so the port's bounded-output
contract holds natively with output_bound = 1.0.

A conformance check (registered below) asserts this stays structurally simple —
no contractive/normalization machinery creeps in — so the cell never quietly
becomes a second Recursion.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .port import SubstratePort, DECAY_CONTROL


class VectorStateCell(nn.Module, SubstratePort):
    CAPABILITIES = frozenset({DECAY_CONTROL})  # read-only decay introspection

    def __init__(self, d_in: int, d_state: int, scalar: float = 0.9):
        nn.Module.__init__(self)
        self.d_in = d_in
        self.d_state = d_state
        self.H = d_state
        self.scalar = float(scalar)     # fixed geometric scalar; not a Parameter
        self.output_bound = 1.0         # tanh wrap ⇒ h ∈ (-1, 1)^H

        self.W_x = nn.Linear(d_in, self.H, bias=True)
        self.W_h = nn.Linear(self.H, self.H, bias=False)
        self.drive_in = nn.Linear(d_state, d_in, bias=False)

        self._h: torch.Tensor | None = None
        self._pending_drive: torch.Tensor | None = None

    # ---- required tier ----------------------------------------------------
    def reset(self, batch_size: int, device=None) -> None:
        # state dtype follows the params (same fix as matrix_cell): fp32 state
        # against bf16 params crashes W_h(self._h) with a mixed-dtype matmul.
        device = device or self.W_x.weight.device
        dtype = self.W_x.weight.dtype
        self._h = torch.zeros(batch_size, self.H, device=device, dtype=dtype)
        self._pending_drive = None

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        if self._h is None:
            self.reset(x_t.shape[0], x_t.device)
        if self._pending_drive is not None:
            x_t = x_t + self.drive_in(self._pending_drive)
            self._pending_drive = None
        self._h = torch.tanh(self.W_x(x_t) + self.scalar * self.W_h(self._h))
        return self._h

    def read(self) -> torch.Tensor:
        if self._h is None:
            raise RuntimeError("VectorStateCell.read() before reset()/step()")
        return self._h

    def write(self, drive: torch.Tensor) -> None:
        self._pending_drive = drive

    def detach_state(self) -> None:
        if self._h is not None:
            self._h = self._h.detach()

    # ---- optional tier ----------------------------------------------------
    def read_decay(self) -> float:
        """READ-ONLY: the fixed geometric scalar. The substrate owns its decay;
        the joiner introspects but never mutates it."""
        return self.scalar


# ---------------------------------------------------------------------------
# Conformance checks (registered for verify_substrate). Structural guards that
# keep this cell simple and distinct from Recursion's contractive machinery.
# ---------------------------------------------------------------------------
from .conformance import make_ast_drift_check, SkipCheck  # noqa: E402

# torch-free: parse this module's source; fail if certificate/contractive
# machinery (spectral norm, carry/EMA, gates, clamping, sigmoid decay) appears,
# and require the disclosed recurrence form.
_vector_simplicity = make_ast_drift_check(
    "aeon.substrate.vector_cell",
    forbidden={"spectral_norm", "carry", "ema", "gate", "forget",
               "clamp", "clip", "sigmoid", "_c"},
    required={"tanh", "W_x", "W_h", "scalar"},
    name="vector_cell_stays_simple",
)


def _vector_runtime_structure(cell):
    """STRUCTURAL (runtime): exactly one state tensor of dim H, no
    gate/carry/forget/in-port/out-port parameter names, a fixed-float scalar,
    and gradient flow through both W_x and W_h."""
    try:
        import torch
    except Exception:
        raise SkipCheck("torch unavailable")
    B = 2
    assert isinstance(cell.scalar, float), "scalar must be a fixed float"
    pnames = dict(cell.named_parameters())
    assert "scalar" not in pnames, "scalar must not be a Parameter"
    for pname in pnames:
        low = pname.lower()
        assert not any(bad in low for bad in
                       ("gate", "carry", "forget", "input", "output")), \
            f"forbidden parameter name: {pname}"
    cell.reset(B)
    cell.step(torch.randn(B, cell.d_in))
    live = {n: t for n, t in vars(cell).items() if isinstance(t, torch.Tensor)}
    assert len(live) == 1, f"expected exactly one state tensor, got {sorted(live)}"
    (state,) = live.values()
    assert tuple(state.shape) == (B, cell.H), f"state shape {tuple(state.shape)} != (B, H)"
    cell.reset(B)
    cell.step(torch.randn(B, cell.d_in))
    loss = cell.step(torch.randn(B, cell.d_in)).sum()
    loss.backward()
    assert cell.W_x.weight.grad is not None, "no gradient into W_x"
    assert (cell.W_h.weight.grad is not None
            and cell.W_h.weight.grad.abs().sum() > 0), "no gradient into W_h"


_vector_runtime_structure.__name__ = "vector_runtime_structure"

VectorStateCell.CONFORMANCE_CHECKS = (_vector_simplicity, _vector_runtime_structure)
