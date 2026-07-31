"""L5 evaluation-only causal interventions.

Every intervention refuses when the model is in training mode. Every
intervention leaves the model, its parameters, and its checkpoints
untouched — it operates on a per-forward alteration hook, not on
persisted state. The same checkpoint / tokenizer / evaluation batches
are used across all intervention identities so results compare
apples-to-apples.

L5 does NOT commit a "demonstrated bypass" claim. It emits per-
intervention deltas; L10 owns the joint decision (predictive
information + causal contribution + barrier selectivity + net
efficiency + stability + repetition).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence, Tuple


class InterventionKind(str, Enum):
    NONE = "NONE"
    ZERO_BROADCAST = "ZERO_BROADCAST"
    FREEZE_BROADCAST = "FREEZE_BROADCAST"
    DELAY_BROADCAST = "DELAY_BROADCAST"
    SHUFFLE_BROADCAST = "SHUFFLE_BROADCAST"
    FREEZE_RECURSION = "FREEZE_RECURSION"
    MASK_TRANSFORMER_SOURCE = "MASK_TRANSFORMER_SOURCE"
    MASK_SUBSTRATE_SOURCE = "MASK_SUBSTRATE_SOURCE"
    NORM_MATCHED_IRRELEVANT_STATE = "NORM_MATCHED_IRRELEVANT_STATE"


@dataclass(frozen=True)
class InterventionSpec:
    kind: InterventionKind
    seed: Optional[int] = None
    delay_windows: int = 0
    parameters: Tuple[Tuple[str, object], ...] = field(default_factory=tuple)


class TrainingModeRefused(RuntimeError):
    """Raised when an intervention is attempted against a model with
    ``model.training=True``."""


def assert_evaluation_mode(model) -> None:
    """Guard called by every intervention entry point."""
    if getattr(model, "training", False):
        raise TrainingModeRefused(
            "Interventions are evaluation-only. "
            "Call model.eval() before intervening.")


def refuses_persistence(fn: Callable) -> Callable:
    """Decorator: an intervention harness function that would ever
    write to a checkpoint dir must NOT be wrapped by this decorator —
    the decorator refuses to accept any keyword named ``checkpoint_dir``.

    Actively checks the call-site for accidental persistence."""
    def wrapper(*args, **kwargs):
        for k in ("checkpoint_dir", "generation_dir", "save_path"):
            if k in kwargs:
                raise RuntimeError(
                    f"L5 intervention refuses persistence keyword {k!r}")
        return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# InterventionRunner — pairs a normal and an intervened forward pass.
# ---------------------------------------------------------------------------
@dataclass
class InterventionResult:
    intervention_identity: str
    baseline_loss: float
    intervened_loss: float
    delta_L_c: float
    barrier_labels: Tuple[bool, ...]
    seed: Optional[int]

    def as_dict(self):
        return {
            "intervention_identity": self.intervention_identity,
            "baseline_loss": self.baseline_loss,
            "intervened_loss": self.intervened_loss,
            "delta_L_c": self.delta_L_c,
            "barrier_labels": list(self.barrier_labels),
            "seed": self.seed,
        }


class InterventionRunner:
    """Coordinates paired normal / intervened evaluations. The runner
    itself is a thin wrapper — the actual per-window intervention hook
    is applied via a caller-supplied closure. This keeps the L5 code
    surface small and gives the closure caller control over exactly
    which tensor is altered (broadcast, recursion, source), without
    modifying HybridModel.forward beyond the L1 observer + reserved
    intervention kwarg.
    """

    def __init__(self, *, model, evaluate_batch: Callable):
        self.model = model
        self._evaluate_batch = evaluate_batch

    @refuses_persistence
    def run(
        self,
        spec: InterventionSpec,
        batches: Sequence,
        *,
        barrier_labels: Sequence[bool] = (),
    ) -> InterventionResult:
        assert_evaluation_mode(self.model)
        # Baseline (no intervention).
        baseline_losses = [self._evaluate_batch(self.model, b, None)
                            for b in batches]
        # Intervened.
        intervened_losses = [self._evaluate_batch(self.model, b, spec)
                              for b in batches]
        # Simple aggregate — L10 does the confidence-interval work.
        baseline = sum(baseline_losses) / max(len(baseline_losses), 1)
        intervened = sum(intervened_losses) / max(len(intervened_losses), 1)
        return InterventionResult(
            intervention_identity=f"{spec.kind.value}:seed={spec.seed}",
            baseline_loss=float(baseline),
            intervened_loss=float(intervened),
            delta_L_c=float(intervened) - float(baseline),
            barrier_labels=tuple(barrier_labels),
            seed=spec.seed,
        )
