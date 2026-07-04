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
        # differentiable per-forward gate accumulator (for L_aux = β·mean g(L))
        self._gate_sum: torch.Tensor | None = None
        self._gate_steps: int = 0
        # optional per-step trajectory recording (for diagnostics T1/T5)
        self.history_enabled: bool = False
        self._load_hist: list[float] = []
        self._gate_hist: list[float] = []
        self._last_base: torch.Tensor | None = None    # last plain readout (diagnostic T3)
        # diagnostic overrides (default OFF — no effect on training):
        self.force_gate: float | None = None           # override g with a constant in [0,1]
        self.inject_noise_std: float = 0.0             # add matched noise to output (T4 floor)

    def reset(self) -> None:
        # per-forward: sensor carry + differentiable gate accumulator. History is
        # NOT cleared here (the diagnostic harness owns it via clear_history()).
        self._prev = self._ewma = self._load = self._gate = None
        self._gate_sum = None
        self._gate_steps = 0

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

        # ---- gate: smooth threshold in [0,1]; keep an fp32 copy for the penalty -
        g_fp32 = torch.sigmoid(self.gate_alpha * (ewma.float() - self.gate_threshold))
        if self.force_gate is not None:              # diagnostic override (T4 / tests)
            g_fp32 = torch.full_like(g_fp32, float(self.force_gate))
        g = g_fp32.to(base.dtype)                                    # (B,) for the blend

        # ---- actuator: convex blend of normal + stressed, bound-preserving ------
        stressed = self.output_bound * torch.tanh(self.W_stressed(base))
        out = (1.0 - g).unsqueeze(-1) * base + g.unsqueeze(-1) * stressed
        if self.inject_noise_std > 0.0:              # diagnostic: matched-noise floor (T4)
            out = (out + self.inject_noise_std * torch.randn_like(out)).clamp(
                -self.output_bound, self.output_bound)

        # differentiable gate accumulator (fp32): L_aux penalises mean firing, so
        # the gate must justify itself by reducing the primary loss.
        step_gate = g_fp32.mean()
        self._gate_sum = step_gate if self._gate_sum is None else self._gate_sum + step_gate
        self._gate_steps += 1

        # carry sensor state as a detached running statistic; stash introspection
        self._prev = base.detach()
        self._ewma = ewma.detach()
        self._load = ewma.detach()
        self._gate = g.detach()
        self._last_base = base.detach()
        if self.history_enabled:
            self._load_hist.append(float(self._load.mean()))
            self._gate_hist.append(float(self._gate.mean()))
        return out

    # ---- introspection (for monitoring, the auxiliary loss, and diagnostics) ---
    def load(self) -> torch.Tensor | None:
        """Last per-batch load L(t) (EWMA of readout rate-of-change), or None."""
        return self._load

    def gate(self) -> torch.Tensor | None:
        """Last per-batch gate value g(L) ∈ [0,1], or None."""
        return self._gate

    def gate_penalty(self) -> torch.Tensor | None:
        """Differentiable mean gate activation over this forward (for L_aux), or
        None if no step ran. Cleared by reset() at the next forward's start."""
        if self._gate_sum is None or self._gate_steps == 0:
            return None
        return self._gate_sum / self._gate_steps

    # ---- trajectory recording (diagnostics own the lifecycle) ------------------
    def enable_history(self) -> None:
        self.history_enabled = True

    def disable_history(self) -> None:
        self.history_enabled = False

    def clear_history(self) -> None:
        self._load_hist = []
        self._gate_hist = []

    def history(self) -> tuple[list[float], list[float]]:
        """(load trajectory, gate trajectory) recorded since the last clear."""
        return list(self._load_hist), list(self._gate_hist)

    def last_base(self) -> torch.Tensor | None:
        """Last plain (pre-gate) readout batch — the normal-path output (T3)."""
        return self._last_base

    def normal_and_stressed(self, base: torch.Tensor):
        """The two output projections on the same states `base` (diagnostic T3):
        normal path (identity — the plain readout) and stressed path."""
        return base, self.output_bound * torch.tanh(self.W_stressed(base))
