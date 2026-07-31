"""L1 authoritative signal trace on HybridModel.forward.

Provides the AeonDiagnosticObserver protocol and the RecursionWindowEvent
dataclass the observer receives.

INVARIANTS (enforced by tests/test_l1_signal_trace.py and
tests/test_ip_preservation.py):

    * When HybridModel.forward is called with observer=None (the
      default), NO code in this module executes. The forward path is
      byte-for-byte identical to a build that never imported
      aeon.bypass.signal_trace.
    * When observer is not None, the observer receives one
      RecursionWindowEvent per K-window boundary crossed inside the
      forward pass. All tensors are .detach()-ed before shape/dtype/
      norm summarisation. Raw tensor values are NEVER captured by
      default; the optional TensorCaptureBudget must be explicitly
      supplied and it fails-closed when the byte or window ceiling is
      exceeded.
    * The observer MUST NOT mutate the model, its parameters, or any
      tensor it inspects.
    * The observer's on_recursion_window callback MUST NOT raise. If it
      does, the diagnostic run aborts loudly — this is a probe bug,
      not a fallback condition.

Namespaces:

    RecursionWindowEvent   — the concrete L1 event dataclass. Superset
                             of the L0 WindowTrace with the full
                             directive-mandated fields.
    AeonDiagnosticObserver — Protocol every probe must satisfy.
    _NullObserver          — internal, used by tests to prove the
                             trivial probe path matches the None path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol, Tuple


# ---------------------------------------------------------------------------
# RecursionWindowEvent — one entry per Aeon K-window boundary
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RecursionWindowEvent:
    """The authoritative summary of one K=16 slow-clock boundary.

    Fields ordered per the L1 directive. Every field is a scalar or
    a bounded tuple. `source_record_ids` is empty by default (raw
    corpus text is never included). `certificate_margin` is None when
    the audit could not be computed at this boundary."""

    schema_version: int
    run_id: str
    checkpoint_generation_id: Optional[str]
    window_index: int
    token_start: int
    token_end: int
    k_value: int

    transformer_source_shape: Tuple[int, ...]
    transformer_source_dtype: str
    transformer_source_norm: float

    substrate_source_shape: Tuple[int, ...]
    substrate_source_dtype: str
    substrate_source_norm: float

    recursion_state_before_shape: Tuple[int, ...]
    recursion_state_before_dtype: str
    recursion_state_before_norm: float

    recursion_state_after_shape: Tuple[int, ...]
    recursion_state_after_dtype: str
    recursion_state_after_norm: float
    recursion_delta_norm: float

    broadcast_shape: Tuple[int, ...]
    broadcast_dtype: str
    broadcast_norm: float

    transformer_consumed_broadcast: bool
    substrate_consumed_broadcast: bool

    certificate_margin: Optional[float]
    source_record_ids: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# AeonDiagnosticObserver — the L1 observer protocol
# ---------------------------------------------------------------------------
class AeonDiagnosticObserver(Protocol):
    """One method per boundary. See module docstring for invariants."""

    def on_recursion_window(self, event: RecursionWindowEvent) -> None:
        ...


# ---------------------------------------------------------------------------
# TensorCaptureBudget — bounded raw-tensor capture policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TensorCaptureBudget:
    """OFFLINE DIAGNOSTIC MODE ONLY. Never write raw tensors to disk
    unless the caller explicitly enables this budget AND provides a
    local sink. `persistent=False` means the sink must be an
    in-memory buffer discarded at the end of the diagnostic run.
    """

    enabled: bool = False
    maximum_bytes: int = 8 * 1024 * 1024   # 8 MiB
    maximum_windows: int = 4
    persistent: bool = False


# ---------------------------------------------------------------------------
# Internal helpers — used ONLY when observer is not None.
# ---------------------------------------------------------------------------
def _shape_of(t) -> Tuple[int, ...]:
    """Return .shape as a plain tuple of Python ints."""
    return tuple(int(d) for d in t.shape)


def _dtype_of(t) -> str:
    return str(t.dtype)


def _norm_of(t) -> float:
    """Detached L2 norm as a Python float."""
    import torch
    with torch.no_grad():
        return float(t.detach().float().norm().item())


def _detached_delta_norm(before, after) -> float:
    """L2 norm of (after - before), detached."""
    import torch
    with torch.no_grad():
        return float((after.detach().float() - before.detach().float()).norm().item())


class _NullObserver:
    """Test-only. Receives events and drops them. Used to prove that
    swapping observer=None for observer=_NullObserver produces
    identical output/gradient/state; if it does not, an unwanted
    side-effect leaked into the observer-active code path."""

    def __init__(self):
        self.events = []

    def on_recursion_window(self, event: RecursionWindowEvent) -> None:
        self.events.append(event)
