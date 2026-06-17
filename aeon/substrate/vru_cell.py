"""
aeon/substrate/vru_cell.py — candidate recurrent substrate, contractive class
(Aeon-original).

A from-scratch contractive recurrent cell realising the "certified, simple"
design archetype contrasted in docs/RWKV_STUDY.md §d — **not** a copy of the
Recursion joiner in recursion.py (that cell *is* the joiner's manifold; this is
a candidate RNN signal source behind the port).

PROVISIONAL. Per the §d information-asymmetry flag, the candidate substrate is
known to this codebase only from a brief port spec; this implementation pins the
*required-tier* behaviour (vector read / vector write / per-token step) plus an
optional decay knob, and uses a spectral-norm-bounded recurrent map to make the
σ<1 contraction explicit. Swap in the real spec when it lands — the port keeps
the joiner unchanged.

State: a hidden vector h and a slow EMA carry c, both width `hidden`. Step:
    h_t = tanh(W_x x_t + margin · W_h h_{t-1})      # W_h spectrally bounded ⇒
                                                     # Lipschitz ≤ margin < 1
    c_t = (1-λ) c_{t-1} + λ h_t                      # per-channel decay λ∈(0,1)
    read = readout(c_t)

NOTE: numeric behaviour is untested in the authoring environment (no torch).
"""
from __future__ import annotations

import torch
import torch.nn as nn

try:  # torch ≥ 2.0 parametrization API
    from torch.nn.utils.parametrizations import spectral_norm
except Exception:  # pragma: no cover - older torch fallback
    from torch.nn.utils import spectral_norm

from .port import SubstratePort, DECAY_CONTROL


class VRUCell(nn.Module, SubstratePort):
    CAPABILITIES = frozenset({DECAY_CONTROL})

    def __init__(
        self,
        d_in: int,
        d_state: int,
        hidden: int | None = None,
        margin: float = 0.95,
    ):
        nn.Module.__init__(self)
        assert 0.0 < margin < 1.0
        self.d_in = d_in
        self.d_state = d_state
        self.hidden = hidden or d_state
        self.margin = margin

        self.W_x = nn.Linear(d_in, self.hidden, bias=True)
        # spectral_norm fixes σ(W_h)=1; the explicit `margin` factor in step()
        # then bounds the recurrent map's Lipschitz constant to margin < 1.
        self.W_h = spectral_norm(nn.Linear(self.hidden, self.hidden, bias=False))
        self.readout = nn.Linear(self.hidden, d_state, bias=False)
        self.drive_in = nn.Linear(d_state, d_in, bias=False)

        # per-channel carry decay λ = sigmoid(lambda_logit) ∈ (0,1)
        self.lambda_logit = nn.Parameter(torch.zeros(self.hidden))

        self._h: torch.Tensor | None = None
        self._c: torch.Tensor | None = None
        self._read: torch.Tensor | None = None
        self._pending_drive: torch.Tensor | None = None
        self._lambda_mod: torch.Tensor | None = None

    def _lambda(self) -> torch.Tensor:
        logit = self.lambda_logit
        if self._lambda_mod is not None:
            logit = logit + self._lambda_mod
        return torch.sigmoid(logit)

    # ---- required tier ----------------------------------------------------
    def reset(self, batch_size: int, device=None) -> None:
        device = device or self.W_x.weight.device
        self._h = torch.zeros(batch_size, self.hidden, device=device)
        self._c = torch.zeros(batch_size, self.hidden, device=device)
        self._read = torch.zeros(batch_size, self.d_state, device=device)
        self._pending_drive = None

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        if self._h is None:
            self.reset(x_t.shape[0], x_t.device)
        if self._pending_drive is not None:
            x_t = x_t + self.drive_in(self._pending_drive)
            self._pending_drive = None

        h = torch.tanh(self.W_x(x_t) + self.margin * self.W_h(self._h))
        lam = self._lambda()
        c = (1.0 - lam) * self._c + lam * h
        self._h, self._c = h, c
        self._read = self.readout(c)
        return self._read

    def read(self) -> torch.Tensor:
        if self._read is None:
            raise RuntimeError("VRUCell.read() before reset()/step()")
        return self._read

    def write(self, drive: torch.Tensor) -> None:
        self._pending_drive = drive

    # ---- optional tier ----------------------------------------------------
    def set_decay(self, mod: torch.Tensor) -> None:
        """Modulate carry decay: λ = sigmoid(lambda_logit + mod). `mod` is
        broadcastable to (hidden,)."""
        self._lambda_mod = mod
