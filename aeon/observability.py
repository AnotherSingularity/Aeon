"""
aeon/observability.py — architecture-preserving instrumentation.

Directive §8: measure the architecture WITHOUT changing it.

Discipline (per §8.3):
  * Always-on metrics are inexpensive scalars — global step, loss, LR, step time,
    tokens/s, certificate status. No device sync beyond what training already does.
  * Sampled metrics (norms, cadences, gate stats) are collected at a configurable
    sparse interval (default: every 512 steps). Sampling on/off does not change
    training semantics.
  * All tensors are `.detach()`-ed and reduced to Python scalars immediately.
  * Logging failure MUST NOT corrupt training — every writer call is guarded.
  * Missing optional metrics do not abort training.

Static accounting (§8.4) — one-shot per configuration:
  * Parameter counts / bytes by top-level component.
  * Optimizer bytes estimate.
  * Recursion state bytes / substrate state bytes.
  * Approximate op counts (STATIC ESTIMATES — not measured FLOPs).

Records land in JSON Lines (`metrics.jsonl`) so downstream tools consume them
without loading torch or the model.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# JSONL writer (guarded, fail-safe)
# ---------------------------------------------------------------------------
class JsonlWriter:
    """Append-only JSONL writer. If the underlying write fails, it swallows the
    error and marks itself broken; subsequent writes are no-ops. Rationale: the
    directive is explicit — logging failure must not corrupt the checkpoint /
    interrupt training."""

    def __init__(self, path: str):
        self.path = path
        self._broken = False
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            # Touch on init so the file exists even when no events fire.
            with open(path, "a", encoding="utf-8"):
                pass
        except Exception:
            self._broken = True

    def write(self, record: Dict[str, Any]) -> None:
        if self._broken:
            return
        try:
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record, default=_json_default) + "\n")
        except Exception:
            self._broken = True                     # never propagate: training must survive


def _json_default(o):
    try:
        if hasattr(o, "item"):
            return o.item()
    except Exception:
        pass
    return str(o)


# ---------------------------------------------------------------------------
# Always-on metrics (inexpensive scalars)
# ---------------------------------------------------------------------------
@dataclass
class AlwaysOnRecord:
    step: int
    loss: float
    lr: float
    step_time_s: float
    tokens_total: int
    useful_tokens_total: int
    tokens_per_s_raw: float
    useful_tokens_per_s: float
    seq_len: int
    resident_mb: float
    recursion_updates_total: int
    certificate_holds: bool
    sigma_h: float
    sigma_c: float
    gamma: float
    non_finite: bool
    checkpoint_status: str = ""
    resume_status: str = ""

    def as_json(self, kind: str = "always_on") -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["kind"] = kind
        return d


# ---------------------------------------------------------------------------
# Sampled metrics (§8.3) — collected at a sparse interval, cheap-ish
# ---------------------------------------------------------------------------
@dataclass
class SampledPhaseTimings:
    substrate_s: float = 0.0
    transformer_s: float = 0.0
    recursion_boundary_s: float = 0.0
    output_loss_s: float = 0.0
    backward_s: float = 0.0
    optimizer_s: float = 0.0
    data_s: float = 0.0


# ---------------------------------------------------------------------------
# Static accounting (§8.4) — one-shot
# ---------------------------------------------------------------------------
def parameter_accounting(model) -> Dict[str, Any]:
    """Parameter counts + bytes broken down by top-level component."""
    top = {}
    total = 0
    total_bytes = 0
    for name, module in model.named_children():
        p = sum(pp.numel() for pp in module.parameters())
        b = sum(pp.numel() * pp.element_size() for pp in module.parameters())
        top[name] = {"parameters": int(p), "bytes": int(b)}
        total += p
        total_bytes += b
    return {
        "total_parameters": int(total),
        "total_bytes": int(total_bytes),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters()
                                        if p.requires_grad)),
        "by_component": top,
        "note": "bytes reflect parameter dtypes at accounting time",
    }


def optimizer_bytes_estimate(model, optimizer_kind: str = "adamw") -> int:
    """Optimizer state overhead estimate. AdamW: 2 fp32 moments per trainable param."""
    if optimizer_kind.lower() == "adamw":
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return int(n * 2 * 4)                       # 2 moments × 4 bytes (fp32)
    return 0


def state_bytes(model) -> Dict[str, int]:
    """Runtime state that isn't a Parameter (approximate)."""
    out = {"recursion_state_bytes": 0, "substrate_state_bytes": 0}
    # Recursion carries h, c per batch — sized at inference; we report a per-B unit
    # bytes-per-element × width. For runtime totals the caller supplies batch.
    rj = getattr(model, "recursion", None)
    if rj is not None:
        h_rec = int(getattr(rj, "H", 0))
        out["recursion_state_bytes_per_batch"] = h_rec * 2 * 4  # h + c, fp32
    sub = getattr(model, "substrate", None)
    if sub is not None:
        # Matrix cell: S ∈ (B, H, N, N) + read (B, d_state); vector: (B, H)
        if hasattr(sub, "H") and hasattr(sub, "N"):
            elem = getattr(sub.receptance.weight, "element_size", lambda: 4)()
            out["substrate_state_bytes_per_batch"] = (
                (sub.H * sub.N * sub.N + sub.d_state) * elem)
        else:
            elem = getattr(sub.W_x.weight, "element_size", lambda: 4)() if hasattr(sub, "W_x") else 4
            out["substrate_state_bytes_per_batch"] = int(getattr(sub, "H", 0)) * elem
    return out


def static_op_estimates(model, seq_len: int, K: int) -> Dict[str, Any]:
    """VERY approximate op counts by top-level component. Labelled clearly as a
    STATIC ESTIMATE, not a hardware-measured FLOP count."""
    # Transformer forward per token ≈ 12·D²·L for attention+MLP feed-forward
    # (rough scaling for a decoder layer with tied embeddings). This is a
    # napkin-quality estimate, and we say so.
    est = {"note": "STATIC ESTIMATE. NOT measured FLOPs."}
    if hasattr(model, "transformer") and hasattr(model.transformer, "cfg"):
        c = model.transformer.cfg
        D = c.hidden_size
        L = c.num_hidden_layers
        F = c.intermediate_size
        est["transformer_ops_per_token_est"] = int(4 * D * D * L + 3 * D * F * L)
        est["transformer_ops_per_forward_est"] = int(est["transformer_ops_per_token_est"] * seq_len)
    if hasattr(model, "recursion"):
        h = int(getattr(model.recursion, "H", 0))
        est["recursion_ops_per_step_est"] = int(4 * h * h)   # order-of-magnitude
        est["recursion_steps_per_forward"] = int(math.ceil(seq_len / K))
    return est


def checkpoint_size_estimate(model) -> int:
    """Bytes of a raw torch.save({'model': state_dict()}) — best proxy."""
    return int(sum(p.numel() * p.element_size() for p in model.parameters())
               + sum(b.numel() * b.element_size() for b in model.buffers()))


# ---------------------------------------------------------------------------
# Observer — the one thing training holds. Non-torch by design so importing
# aeon.observability is cheap and torch-free where possible.
# ---------------------------------------------------------------------------
class Observer:
    """Manages metrics for a training run.

    Design:
      * always-on emit is a scalar-only append; expected cost ≪ one step.
      * sampled emit is gated by step % sample_every == 0. Between samples the
        observer records only wall-clock; no phase timers run.
      * `disable()` fully disables emission (equivalent to no-op). The
        instrumentation-equivalence tests exercise this: sampling on/off must not
        change model outputs, gradients, or optimizer updates.
      * every emission is guarded — an exception during writing broke this event
        but the next event still tries.
    """

    def __init__(self, out_dir: str, sample_every: int = 512, enabled: bool = True):
        self.enabled = enabled
        self.sample_every = int(sample_every)
        self._writer = JsonlWriter(os.path.join(out_dir, "metrics.jsonl"))
        self._t0 = time.time()
        self._prev_step_wall = None
        # cumulative counters (always-on inputs)
        self._tokens_total = 0
        self._useful_tokens_total = 0
        self._recursion_updates_total = 0
        # sampled-timing scratch
        self._phase = SampledPhaseTimings()

    # ---- controls ----------------------------------------------------------
    def disable(self):
        self.enabled = False

    def enable(self):
        self.enabled = True

    def should_sample(self, step: int) -> bool:
        return self.enabled and self.sample_every > 0 and (step % self.sample_every == 0)

    # ---- accumulators used from the training loop -------------------------
    def add_tokens(self, tokens: int, useful_tokens: int):
        self._tokens_total += int(tokens)
        self._useful_tokens_total += int(useful_tokens)

    def add_recursion_updates(self, n: int):
        self._recursion_updates_total += int(n)

    # ---- phase timers (very cheap; used only inside sampled windows) -------
    def _mark(self, key: str, dt: float):
        setattr(self._phase, key, getattr(self._phase, key, 0.0) + float(dt))

    def phase(self, name: str):
        obs = self

        class _Ctx:
            def __enter__(_):
                _._t = time.perf_counter()
                return _

            def __exit__(_, *args):
                obs._mark(f"{name}_s", time.perf_counter() - _._t)
                return False
        return _Ctx()

    # ---- emitters ---------------------------------------------------------
    def emit_always_on(self, **fields):
        if not self.enabled:
            return
        rec = AlwaysOnRecord(
            step=fields["step"], loss=float(fields["loss"]), lr=float(fields["lr"]),
            step_time_s=float(fields["step_time_s"]),
            tokens_total=self._tokens_total,
            useful_tokens_total=self._useful_tokens_total,
            tokens_per_s_raw=float(fields.get("tokens_per_s_raw", 0.0)),
            useful_tokens_per_s=float(fields.get("useful_tokens_per_s", 0.0)),
            seq_len=int(fields["seq_len"]),
            resident_mb=float(fields.get("resident_mb", 0.0)),
            recursion_updates_total=self._recursion_updates_total,
            certificate_holds=bool(fields["certificate_holds"]),
            sigma_h=float(fields["sigma_h"]),
            sigma_c=float(fields["sigma_c"]),
            gamma=float(fields["gamma"]),
            non_finite=bool(fields.get("non_finite", False)),
            checkpoint_status=str(fields.get("checkpoint_status", "")),
            resume_status=str(fields.get("resume_status", "")),
        )
        self._writer.write(rec.as_json())

    def emit_sampled(self, step: int, **fields):
        """Emit sampled metrics. `fields` accepts detached scalars only."""
        if not self.enabled:
            return
        rec = {"kind": "sampled", "step": int(step), "phase_s": self._phase.__dict__.copy()}
        rec.update({k: (v.item() if hasattr(v, "item") else v) for k, v in fields.items()})
        self._writer.write(rec)
        # reset phase scratch after emitting
        self._phase = SampledPhaseTimings()

    def emit_static(self, kind: str, payload: Dict[str, Any]):
        if not self.enabled:
            return
        rec = {"kind": kind, **payload}
        self._writer.write(rec)


# ---------------------------------------------------------------------------
# Resident-memory helper (best-effort; safe if /proc missing).
# ---------------------------------------------------------------------------
def resident_mb() -> float:
    try:
        with open("/proc/self/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    kb = int(line.split()[1])
                    return kb / 1024.0
    except Exception:
        pass
    return 0.0
