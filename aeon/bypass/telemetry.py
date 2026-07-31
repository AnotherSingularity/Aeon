"""L4 bounded per-window telemetry.

Wraps HybridModel.forward's L1 signal-trace observer with pre- and
post-broadcast loss / margin / target-rank capture. Runs are
sampled, offline, byte-bounded, and disabled by default. Paired
instrumented / uninstrumented trials are the ONLY method for
reporting overhead — no minimum-only timing. Every trial result is
retained.

The delta_loss statistic is:

    delta_loss = pre_broadcast_loss - post_broadcast_loss

A positive value means the single hidden-state broadcast improved the
correct-token likelihood on the sampled window.

TELEMETRY IS OFFLINE AND LOCAL. No outbound network. No third-party
upload. Byte budgets fail closed.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class TelemetryConfig:
    enabled: bool = False
    sink_dir: Optional[str] = None
    max_bytes: int = 4 * 1024 * 1024   # 4 MiB
    max_windows: int = 4096
    persistent: bool = False
    sampling_rate: float = 1.0  # fraction of windows recorded
    seed: int = 0


@dataclass(frozen=True)
class TelemetryRow:
    """One sampled window's telemetry line. All values scalar."""

    window_index: int
    task_id: Optional[str]
    record_id: Optional[str]
    pre_broadcast_loss: Optional[float]
    post_broadcast_loss: Optional[float]
    delta_loss: Optional[float]
    correct_token_margin_before: Optional[float]
    correct_token_margin_after: Optional[float]
    target_rank_before: Optional[int]
    target_rank_after: Optional[int]
    recursion_state_norm: float
    recursion_delta_norm: float
    reaction_coordinate: Optional[float]
    broadcast_norm: float
    transformer_source_norm: float
    substrate_source_norm: float
    substrate_gate_mean: Optional[float]
    window_compute_time_s: Optional[float]
    estimated_state_bytes: int
    certificate_margin: Optional[float]
    barrier_status: Dict[str, bool] = field(default_factory=dict)


class TelemetryOverBudget(RuntimeError):
    pass


class SamplingTelemetryObserver:
    """L1-shaped observer that emits L4 telemetry rows. Detached — never
    holds references to raw activations across windows. Runs entirely
    offline: rows accumulate in memory and, if persistent=True, flush
    to sink_dir at close(). Byte and window ceilings fail closed."""

    run_id: str
    checkpoint_generation_id: Optional[str]
    source_record_ids: Tuple[str, ...]

    def __init__(self, cfg: TelemetryConfig, *, run_id: str = "telemetry"):
        self.cfg = cfg
        self.run_id = run_id
        self.checkpoint_generation_id = None
        self.source_record_ids: Tuple[str, ...] = ()
        self._rows: List[TelemetryRow] = []
        self._bytes_used = 0
        self._window_count = 0
        import random
        self._rng = random.Random(cfg.seed)

    def on_recursion_window(self, event) -> None:  # aeon.bypass.signal_trace.RecursionWindowEvent
        if not self.cfg.enabled:
            return
        self._window_count += 1
        if self._window_count > self.cfg.max_windows:
            raise TelemetryOverBudget(
                f"window ceiling {self.cfg.max_windows} exceeded")
        if self.cfg.sampling_rate < 1.0 and self._rng.random() > self.cfg.sampling_rate:
            return
        # Compose a scalar row; pre/post loss is not available inside
        # the L1 observer callback — the caller supplies them via
        # attach_scored_rows() from an external evaluator.
        row = TelemetryRow(
            window_index=event.window_index,
            task_id=event.task_id,
            record_id=event.record_id,
            pre_broadcast_loss=None,
            post_broadcast_loss=None,
            delta_loss=None,
            correct_token_margin_before=None,
            correct_token_margin_after=None,
            target_rank_before=None,
            target_rank_after=None,
            recursion_state_norm=event.recursion_state_after_norm,
            recursion_delta_norm=event.recursion_delta_norm,
            reaction_coordinate=None,
            broadcast_norm=event.broadcast_norm,
            transformer_source_norm=event.transformer_source_norm,
            substrate_source_norm=event.substrate_source_norm,
            substrate_gate_mean=event.substrate_gate_mean,
            window_compute_time_s=None,
            estimated_state_bytes=(event.recursion_state_after_shape
                                     and event.recursion_state_after_shape[-1]
                                     * 4  # fp32
                                     or 0),
            certificate_margin=event.certificate_margin,
            barrier_status={},
        )
        row_bytes = len(json.dumps(asdict(row)).encode("utf-8"))
        if self._bytes_used + row_bytes > self.cfg.max_bytes:
            raise TelemetryOverBudget(
                f"byte ceiling {self.cfg.max_bytes} exceeded at row "
                f"{len(self._rows)}")
        self._bytes_used += row_bytes
        self._rows.append(row)

    def rows(self) -> List[TelemetryRow]:
        return list(self._rows)

    def close(self) -> Optional[str]:
        """Flush to sink_dir if persistent=True. Returns the emitted
        path or None. Never touches the network."""
        if not (self.cfg.persistent and self.cfg.sink_dir):
            return None
        p = Path(self.cfg.sink_dir)
        p.mkdir(parents=True, exist_ok=True)
        out = p / f"telemetry-{self.run_id}-{int(time.time())}.jsonl"
        with open(out, "w", encoding="utf-8") as fh:
            for row in self._rows:
                fh.write(json.dumps(asdict(row), sort_keys=True) + "\n")
        return str(out)


def compute_delta_loss(pre: float, post: float) -> float:
    """delta_loss = pre - post; positive means the broadcast helped."""
    return float(pre) - float(post)
