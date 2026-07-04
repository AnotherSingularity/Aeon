"""
aeon/substrate/feedback.py — substrate adaptive feedback control.

Turns a substrate readout into a closed-loop control signal. Under normal load
the loop is open (output == the plain readout); under stress the loop engages and
the output is transformed to be sharper / more directional, which Recursion
integrates and broadcasts back to both streams to relieve the load.

Three pieces (the sensor / gate / actuator of a control loop):

  * SENSOR — a scalar load L(t) per batch element: an EWMA of the readout's
    per-step rate of change ``mean|base_t - base_{t-1}|``. Cheap, and bounded in
    [0, 2·output_bound] because the readout is bounded. The EWMA (short memory)
    de-noises transient spikes so the mode does not chatter.

  * GATE — g(L) = sigmoid(α·(L − θ)) ∈ [0, 1]: smooth (differentiable for
    training), flat when load is low, sharp at the θ threshold, saturating at 1
    under high load. α, θ are learned fp32 master parameters (a learned θ≈0.5 in
    bf16 has ULP above the optimizer step and would freeze — the same trap γ hit,
    so these stay fp32 and are re-cast after any global dtype cast).

  * ACTUATOR — a stressed transform of the readout, blended by the gate:
        output = (1 − g)·base + g·(output_bound · tanh(W_stressed · base))
    Both terms are bounded by output_bound and g ∈ [0, 1], so the convex blend is
    bounded by output_bound ELEMENTWISE — the port's bounded-output contract and
    hence Recursion's system-wide certificate hold in every mode. What changes
    under stress is the DIRECTION/structure of the output, never its magnitude.

W_stressed is randomly initialised (std 0.02, like the write_proj patch) so it is
a meaningfully different projection, not a trivial copy of the normal path. With
g≈0 at low load the extension reduces cleanly to the pre-extension readout.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class AdaptiveFeedbackController(nn.Module):
    """Load-sensing gated feedback over a bounded substrate readout.

    Call `reset()` at each sequence start and `detach()` at truncated-BPTT window
    boundaries (the carried sensor state is a running statistic, detached so it
    never grows the graph). `forward(base)` maps the plain readout `base`
    (|base| ≤ output_bound) to the gated output (same bound)."""

    def __init__(
        self,
        d_state: int,
        output_bound: float = 1.0,
        alpha_init: float = 8.0,
        threshold_init: float = 0.5,
        ewma_rho: float = 0.9,
        learn_gate: bool = True,
        stressed_std: float = 0.02,
    ):
        super().__init__()
        self.d_state = d_state
        self.output_bound = float(output_bound)
        self.ewma_rho = float(ewma_rho)

        # actuator: the stressed-mode output projection (new; normal path is the
        # caller's existing readout, so gate-off == current behaviour exactly).
        self.W_stressed = nn.Linear(d_state, d_state, bias=False)
        nn.init.normal_(self.W_stressed.weight, std=stressed_std)

        # gate parameters — fp32 master scalars (see module docstring). Learned by
        # default; fixed buffers if learn_gate=False.
        a = torch.tensor(float(alpha_init), dtype=torch.float32)
        t = torch.tensor(float(threshold_init), dtype=torch.float32)
        if learn_gate:
            self.gate_alpha = nn.Parameter(a)
            self.gate_threshold = nn.Parameter(t)
        else:
            self.register_buffer("gate_alpha", a)
            self.register_buffer("gate_threshold", t)

        # carried sensor state + last-step introspection (not parameters)
        self._prev: torch.Tensor | None = None
        self._ewma: torch.Tensor | None = None
        self._load: torch.Tensor | None = None
        self._gate: torch.Tensor | None = None

    def reset(self) -> None:
        self._prev = self._ewma = self._load = self._gate = None

    def detach(self) -> None:
        if self._prev is not None:
            self._prev = self._prev.detach()
        if self._ewma is not None:
            self._ewma = self._ewma.detach()

    def forward(self, base: torch.Tensor) -> torch.Tensor:
        # ---- sensor: EWMA of the readout's per-step rate of change (bounded) ---
        if self._prev is None:
            delta = torch.zeros(base.shape[0], device=base.device, dtype=base.dtype)
        else:
            delta = (base - self._prev).abs().mean(dim=-1)          # (B,)
        ewma = delta if self._ewma is None else (
            self.ewma_rho * self._ewma + (1.0 - self.ewma_rho) * delta)

        # ---- gate: smooth threshold in [0,1], computed in fp32 then cast --------
        g = torch.sigmoid(self.gate_alpha * (ewma.float() - self.gate_threshold))
        g = g.to(base.dtype)                                         # (B,)

        # ---- actuator: convex blend of normal + stressed, bound-preserving ------
        stressed = self.output_bound * torch.tanh(self.W_stressed(base))
        out = (1.0 - g).unsqueeze(-1) * base + g.unsqueeze(-1) * stressed

        # carry sensor state as a detached running statistic; stash introspection
        self._prev = base.detach()
        self._ewma = ewma.detach()
        self._load = ewma.detach()
        self._gate = g.detach()
        return out

    # ---- introspection (for monitoring and a possible auxiliary loss) ----------
    def load(self) -> torch.Tensor | None:
        """Last per-batch load L(t) (EWMA of readout rate-of-change), or None."""
        return self._load

    def gate(self) -> torch.Tensor | None:
        """Last per-batch gate value g(L) ∈ [0,1], or None."""
        return self._gate
