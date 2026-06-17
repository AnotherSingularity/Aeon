"""
aeon/substrate/rwkv_cell.py — RWKV-class substrate (Aeon-original).

A from-scratch implementation of the RWKV design *archetype* studied in
docs/RWKV_STUDY.md — **not** a wrapper or import of BlinkDL/RWKV-LM. It realises
the matrix-state, per-channel-decay, outer-product-write recurrence as an
`nn.Module` that satisfies `SubstratePort`, and advertises the rich optional
ports (matrix read, decay control, delta-rule association write) that make the
RWKV archetype worth deploying as the RNN signal source.

State (per batch element): a matrix S of shape (H, N, N) — H heads of size N.
Recurrence (RNN form, v6-style core):
    a_t   = k_t ⊗ v_t                      # outer-product write   (H, N, N)
    S_t   = a_t + diag(w) · S_{t-1}        # per-channel decay w∈(0,1)
    out_t = r_t · S_t                       # receptance read       (H, 1, N)
The v7 delta-rule transition (state @ ab erase term) is a documented extension
point on top of this core; see `assoc_write` and the note in `step`.

NOTE: numeric behaviour is untested in the authoring environment (no torch
available there); the conformance test in tests/ exercises it where torch is
installed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .port import SubstratePort, MATRIX_READ, DECAY_CONTROL, ASSOC_WRITE


class RWKVCell(nn.Module, SubstratePort):
    CAPABILITIES = frozenset({MATRIX_READ, DECAY_CONTROL, ASSOC_WRITE})

    def __init__(
        self,
        d_in: int,
        d_state: int,
        n_head: int = 4,
        head_size: int = 16,
    ):
        nn.Module.__init__(self)
        self.d_in = d_in
        self.d_state = d_state
        self.H = n_head
        self.N = head_size
        self.dim_att = n_head * head_size

        # projections: input -> receptance / key / value
        self.receptance = nn.Linear(d_in, self.dim_att, bias=False)
        self.key = nn.Linear(d_in, self.dim_att, bias=False)
        self.value = nn.Linear(d_in, self.dim_att, bias=False)
        # flattened per-head readout -> d_state
        self.readout = nn.Linear(self.dim_att, d_state, bias=False)
        # joiner drive (d_state) -> input space (d_in), for write()
        self.drive_in = nn.Linear(d_state, d_in, bias=False)

        # per-channel decay, stored as a logit; w = sigmoid(logit) ∈ (0,1).
        # Init spread across timescales (fast→slow), echoing RWKV's decay init.
        lin = torch.linspace(-4.0, 4.0, self.dim_att)
        self.decay_logit = nn.Parameter(lin)

        # runtime state (not parameters)
        self._S: torch.Tensor | None = None          # (B, H, N, N)
        self._read: torch.Tensor | None = None        # (B, d_state)
        self._pending_drive: torch.Tensor | None = None
        self._decay_mod: torch.Tensor | None = None    # optional (dim_att,) logit shift

    # ---- helpers ----------------------------------------------------------
    def _w(self) -> torch.Tensor:
        logit = self.decay_logit
        if self._decay_mod is not None:
            logit = logit + self._decay_mod
        return torch.sigmoid(logit).view(self.H, self.N)  # (H, N)

    # ---- required tier ----------------------------------------------------
    def reset(self, batch_size: int, device=None) -> None:
        device = device or self.receptance.weight.device
        self._S = torch.zeros(batch_size, self.H, self.N, self.N, device=device)
        self._read = torch.zeros(batch_size, self.d_state, device=device)
        self._pending_drive = None

    def step(self, x_t: torch.Tensor) -> torch.Tensor:
        if self._S is None:
            self.reset(x_t.shape[0], x_t.device)
        if self._pending_drive is not None:
            x_t = x_t + self.drive_in(self._pending_drive)
            self._pending_drive = None

        B = x_t.shape[0]
        H, N = self.H, self.N
        k = self.key(x_t).view(B, H, N, 1)
        v = self.value(x_t).view(B, H, 1, N)
        r = self.receptance(x_t).view(B, H, 1, N)

        a = k @ v                                   # (B, H, N, N) outer product
        w = self._w().view(1, H, N, 1)              # decay on the key axis
        self._S = a + w * self._S                   # per-channel decayed accumulate
        # (v7 delta-rule erase term would subtract S @ ab here; see assoc_write)

        out = (r @ self._S).reshape(B, H * N)       # receptance read, flatten heads
        self._read = self.readout(out)
        return self._read

    def read(self) -> torch.Tensor:
        if self._read is None:
            raise RuntimeError("RWKVCell.read() before reset()/step()")
        return self._read

    def write(self, drive: torch.Tensor) -> None:
        self._pending_drive = drive

    # ---- optional tier ----------------------------------------------------
    def read_matrix(self) -> torch.Tensor:
        """Raw (B, H, N, N) state — the rich port a joiner can cross-attend into."""
        if self._S is None:
            raise RuntimeError("RWKVCell.read_matrix() before reset()/step()")
        return self._S

    def set_decay(self, mod: torch.Tensor) -> None:
        """Modulate per-channel decay: w = sigmoid(decay_logit + mod). `mod` is
        broadcastable to (dim_att,)."""
        self._decay_mod = mod

    def assoc_write(self, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor | None = None) -> None:
        """Delta-rule-style association write: S += scale · (k ⊗ v).

        k, v are (B, H, N); `a` (optional, broadcastable) scales the write — the
        v7 "in-context learning rate". This is the hook through which a closed
        loop drives associations into the substrate (see §e-B)."""
        if self._S is None:
            raise RuntimeError("RWKVCell.assoc_write() before reset()/step()")
        B, H, N = k.shape
        outer = k.view(B, H, N, 1) @ v.view(B, H, 1, N)
        if a is not None:
            outer = outer * a.view(B, H, 1, 1)
        self._S = self._S + outer
