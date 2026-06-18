"""
aeon/substrate/matrix_cell.py — Aeon matrix-state recurrent cell.

A signal-source cell that satisfies the substrate port: per batch element it
carries a matrix state S of shape (H, N, N) — H heads of size N — evolved by
per-channel decay plus an outer-product write, and read by a receptance vector.

Recurrence (per token):
    a_t   = k_t ⊗ v_t                      # outer-product write   (H, N, N)
    S_t   = a_t + diag(w) · S_{t-1}        # per-channel decay w ∈ (0,1)
    out_t = r_t · S_t                       # receptance read

Bounded output: the readout is tanh-wrapped ⇒ output ∈ (-1, 1)^d_state,
output_bound = 1.0 (the port requires bounded readouts).

Optional ports advertised: matrix_read (raw S), decay_control (read-only decay),
assoc_write (direct association write). Optional RMSNorm on the read path is
available (off by default) for state-magnitude control at scale.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .port import SubstratePort, MATRIX_READ, DECAY_CONTROL, ASSOC_WRITE


class MatrixStateCell(nn.Module, SubstratePort):
    CAPABILITIES = frozenset({MATRIX_READ, DECAY_CONTROL, ASSOC_WRITE})

    def __init__(
        self,
        d_in: int,
        d_state: int,
        n_head: int = 4,
        head_size: int = 16,
        use_state_norm: bool = False,
    ):
        nn.Module.__init__(self)
        self.d_in = d_in
        self.d_state = d_state
        self.output_bound = 1.0
        self.H = n_head
        self.N = head_size
        self.dim_att = n_head * head_size

        # optional RMSNorm on the per-head state before the read (read path only;
        # the stored accumulator keeps its raw decay/accumulation). Off by
        # default; a debug knob for matrix-state magnitude drift at scale.
        self.use_state_norm = use_state_norm
        if use_state_norm:
            self.state_norm_weight = nn.Parameter(torch.ones(head_size))
            self.state_norm_eps = 1e-5

        self.receptance = nn.Linear(d_in, self.dim_att, bias=False)
        self.key = nn.Linear(d_in, self.dim_att, bias=False)
        self.value = nn.Linear(d_in, self.dim_att, bias=False)
        self.readout = nn.Linear(self.dim_att, d_state, bias=False)
        self.drive_in = nn.Linear(d_state, d_in, bias=False)

        # per-channel decay, stored as a logit; w = sigmoid(logit) ∈ (0,1). Init
        # spread across timescales (fast -> slow). Substrate-owned: no external
        # decay mutator (decay is read-only).
        self.decay_logit = nn.Parameter(torch.linspace(-4.0, 4.0, self.dim_att))

        self._S: torch.Tensor | None = None
        self._read: torch.Tensor | None = None
        self._pending_drive: torch.Tensor | None = None

    def _w(self) -> torch.Tensor:
        return torch.sigmoid(self.decay_logit).view(self.H, self.N)

    # ---- required tier ----------------------------------------------------
    def reset(self, batch_size: int, device=None) -> None:
        # state dtype must follow the params; fp32 state against bf16 params
        # crashes `r @ S` with a mixed-dtype matmul error.
        device = device or self.receptance.weight.device
        dtype = self.receptance.weight.dtype
        self._S = torch.zeros(batch_size, self.H, self.N, self.N, device=device, dtype=dtype)
        self._read = torch.zeros(batch_size, self.d_state, device=device, dtype=dtype)
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
        w = self._w().view(1, H, N, 1)
        self._S = a + w * self._S                   # per-channel decayed accumulate

        S_read = self._S
        if self.use_state_norm:
            S_read = (S_read * torch.rsqrt(S_read.pow(2).mean(-1, keepdim=True)
                                           + self.state_norm_eps)
                      * self.state_norm_weight)
        out = (r @ S_read).reshape(B, H * N)
        self._read = torch.tanh(self.readout(out))  # bounded-output contract: (-1,1)
        return self._read

    def read(self) -> torch.Tensor:
        if self._read is None:
            raise RuntimeError("MatrixStateCell.read() before reset()/step()")
        return self._read

    def write(self, drive: torch.Tensor) -> None:
        self._pending_drive = drive

    def detach_state(self) -> None:
        if self._S is not None:
            self._S = self._S.detach()
        if self._read is not None:
            self._read = self._read.detach()

    # ---- optional tier ----------------------------------------------------
    def read_matrix(self) -> torch.Tensor:
        """Raw (B, H, N, N) state — the rich port a joiner can read into."""
        if self._S is None:
            raise RuntimeError("MatrixStateCell.read_matrix() before reset()/step()")
        return self._S

    def read_decay(self) -> torch.Tensor:
        """READ-ONLY per-channel decay w = sigmoid(decay_logit), shape (H, N)."""
        return self._w()

    def assoc_write(self, k: torch.Tensor, v: torch.Tensor, a: torch.Tensor | None = None) -> None:
        """Association write: S += a · (k ⊗ v). k, v are (B, H, N); a (optional,
        broadcastable) scales the write. The substrate's write port for a loop."""
        if self._S is None:
            raise RuntimeError("MatrixStateCell.assoc_write() before reset()/step()")
        B, H, N = k.shape
        outer = k.view(B, H, N, 1) @ v.view(B, H, 1, N)
        if a is not None:
            outer = outer * a.view(B, H, 1, 1)
        self._S = self._S + outer
