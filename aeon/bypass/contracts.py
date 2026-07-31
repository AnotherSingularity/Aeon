"""L0 protocols and dataclasses for the latent-bypass instrumentation.

The types declared here are the frozen contract every downstream
L-series module consumes. Written at L0 so redefinition after
observation is visibly a code change, not a subtle drift in an
intermediate module.

Everything here is evaluation-only. No type here is imported by, or
observed from, `HybridModel.forward` in its default (probe=None,
intervention=None) path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, Tuple


# ---------------------------------------------------------------------------
# WindowTrace — one entry per Recursion K=16 boundary
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class WindowTrace:
    """A single slow-clock boundary's summary statistics.

    Every field is a scalar. Raw tensor capture is a separate opt-in
    facility (see `TensorCaptureBudget`) with a strict byte budget so a
    diagnostic run cannot silently balloon into a full dump.
    """

    window_index: int
    token_start: int
    token_end: int

    # Digests of the pre-broadcast and post-broadcast transformer states.
    # Used to detect noninterference violations across runs — if two
    # runs with identical inputs produce different pre-broadcast digests,
    # the probe is not detached / free of side effects.
    transformer_pre_broadcast_digest: str
    transformer_post_broadcast_digest: str

    # Norm summaries — cheap and always emitted.
    transformer_input_norm: float
    substrate_input_norm: float
    recursion_state_before_norm: float
    recursion_state_after_norm: float
    recursion_delta_norm: float
    broadcast_norm: float

    # Certificate margin at this boundary. The isolated contractive
    # certificate is a training-time invariant; here it is observed
    # per-window so L9 stability analysis can regress it against loop
    # gain estimates.
    certificate_margin: float

    # Optional substrate-side signal — None when the substrate does not
    # emit a gate this window.
    substrate_gate_mean: Optional[float]

    # Task metadata (populated by the caller if known). None when the
    # probe is running outside a task-labeled evaluation.
    task_id: Optional[str] = None
    record_id: Optional[str] = None


# ---------------------------------------------------------------------------
# BypassProbe — the ONE callback surface HybridModel.forward exposes
# ---------------------------------------------------------------------------
class BypassProbe(Protocol):
    """Optional observer surface passed into `HybridModel.forward` at L1.

    A probe MUST:

    * Detach every tensor it inspects before summarising it.
    * Not retain references to raw activations across `on_window`.
    * Not mutate any model parameter or module.
    * Complete each call in bounded time and memory.
    * Tolerate being called zero times (short input) or many times.

    A probe MAY:

    * Accumulate scalar summaries.
    * Write structured events to disk.
    * Feed downstream L-series estimators.
    """

    def on_window(self, event: WindowTrace) -> None:
        """Receive one window's summary. Must not raise; a probe that
        does raise will cause the diagnostic evaluation to abort loudly
        rather than degrade silently, but that is a bug in the probe."""
        ...


# ---------------------------------------------------------------------------
# EvaluationIntervention — L5 causal-intervention envelope
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluationIntervention:
    """An evaluation-only alteration to the broadcast pathway.

    The concrete intervention kinds land at L5. At L0 the envelope
    exists so `HybridModel.forward` can accept the argument with a
    stable type without waiting for L5. `kind` is a string here, not
    an Enum, so L5 can introduce its enum without breaking the L0
    contract; the training-time guard belongs in the caller.
    """

    kind: str
    seed: Optional[int] = None
    parameters: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# TensorCaptureBudget — bounded raw-tensor capture policy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TensorCaptureBudget:
    """Policy for raw-tensor capture inside a diagnostic run.

    Off by default. When enabled, the probe layer will refuse to
    accumulate additional raw tensors after `max_bytes` and record the
    truncation in evidence. A probe that ignores this budget is a
    bug — evaluators MUST respect it so a diagnostic run cannot
    dominate disk.
    """

    enabled: bool = False
    max_bytes: int = 0
    max_windows: int = 0
    dtype_hint: str = "float32"


# ---------------------------------------------------------------------------
# BarrierDefinition — L2 registry entry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BarrierDefinition:
    """One entry in the L2 visible barrier registry.

    Barriers are computed from VISIBLE data only. A definition binds a
    metric name (e.g., "pre_broadcast_nll") to a calibration policy
    (e.g., "top_5_percent_on_calibration") and a compatible task set.
    """

    barrier_id: str
    version: int
    metric: str
    threshold_policy: str
    calibration_partition: str
    evaluation_partition: str
    pre_broadcast: bool
    applicable_tasks: Tuple[str, ...]


# ---------------------------------------------------------------------------
# CorpusPartitionManifest — L3+ real-corpus staging record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CorpusPartitionManifest:
    """Metadata for a vendored L3+ real-English partition.

    Every L3+ measurement records the specific manifest instance that
    produced its data. `test_sealed_until` is a human-readable ISO date
    or the string "sealed"; L11 refuses to publish held-out results if
    the seal is lifted before thresholds and reaction coordinates are
    locked.
    """

    manifest_id: str
    source_identity: str
    public_domain_attestation: str
    retrieved_at: str
    file_digest_sha256: str
    preprocessing_version: int
    tokenizer_identity: str
    partition: str  # "train" | "calibration" | "validation" | "test"
    record_count: int
    total_tokens: int
    test_sealed_until: Optional[str] = None
