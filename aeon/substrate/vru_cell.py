"""
aeon/substrate/vru_cell.py — candidate recurrent substrate (Aeon-original).

The disclosed candidate-substrate spec. This is NOT Recursion's mechanism: no
spectral-norm bound, no σ<1 certificate, no second state stream, no decay/gate
streams. A single state, a fixed geometric scalar, a tanh wrap. The conformance
tests assert these properties structurally so the implementation cannot drift
back into Recursion-class.

Spec
----
  state      : a single tensor h of dimension H (= d_state)
  recurrence : h_new = tanh(W_x @ x + scalar * W_h @ h)
  scalar     : a fixed geometric constant (not learned, not clamped)
  no gates, no carry stream, no split state

Output is the state h itself. Because h = tanh(...), every readout lies in
(-1, 1)^H by construction, so the port's bounded-output contract holds natively
with output_bound = 1.0.

NOTE: numeric behaviour is untested in the authoring environment (no torch);
the conformance tests exercise it where torch is installed.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .port import SubstratePort, DECAY_CONTROL


class VRUCell(nn.Module, SubstratePort):
    CAPABILITIES = frozenset({DECAY_CONTROL})  # read-only decay introspection

    def __init__(self, d_in: int, d_state: int, scalar: float = 0.9):
        nn.Module.__init__(self)
        self.d_in = d_in
        self.d_state = d_state          # the state h has dimension H = d_state
        self.H = d_state
        self.scalar = float(scalar)     # fixed geometric scalar; not a Parameter
        self.output_bound = 1.0         # tanh wrap ⇒ h ∈ (-1, 1)^H

        self.W_x = nn.Linear(d_in, self.H, bias=True)
        self.W_h = nn.Linear(self.H, self.H, bias=False)
        # joiner drive (d_state) -> input space (d_in), for the write port
        self.drive_in = nn.Linear(d_state, d_in, bias=False)

        self._h: torch.Tensor | None = None            # the single state tensor
        self._pending_drive: torch.Tensor | None = None

    # ---- required tier ----------------------------------------------------
    def reset(self, batch_size: int, device=None) -> None:
        device = device or self.W_x.weight.device
        self._h = torch.zeros(batch_size, self.H, device=device)
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
            raise RuntimeError("VRUCell.read() before reset()/step()")
        return self._h

    def write(self, drive: torch.Tensor) -> None:
        self._pending_drive = drive

    # ---- optional tier ----------------------------------------------------
    def read_decay(self) -> float:
        """READ-ONLY: the fixed geometric scalar. The substrate owns its decay;
        the joiner introspects but never mutates it."""
        return self.scalar


# ---------------------------------------------------------------------------
# Per-cell conformance checks, registered for verify_substrate(). These extend
# the port suite with VRU-specific structural guards without forking the utility.
# ---------------------------------------------------------------------------
from .conformance import make_ast_drift_check, SkipCheck  # noqa: E402

# structural anti-drift (torch-free): parses this module's source and fails if
# Recursion-class mechanisms reappear; asserts the disclosed recurrence form.
_vru_anti_drift = make_ast_drift_check(
    "aeon.substrate.vru_cell",
    forbidden={"spectral_norm", "carry", "ema", "gate", "forget",
               "clamp", "clip", "sigmoid", "_c"},
    required={"tanh", "W_x", "W_h", "scalar"},
    name="vru_no_recursion_class_mechanisms",
)


def _vru_runtime_structure(cell):
    """STRUCTURAL (runtime): exactly one state tensor of dim H, no
    gate/carry/forget/in-port/out-port parameter names, a fixed-float scalar,
    and gradient flow through both W_x and W_h — verified structurally, not by
    numeric comparison to reference outputs."""
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


_vru_runtime_structure.__name__ = "vru_runtime_structure"

VRUCell.CONFORMANCE_CHECKS = (_vru_anti_drift, _vru_runtime_structure)
