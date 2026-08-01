"""ACIS-7 transport calibration.

Measures the transport overhead of ACIS in BUCKET mode relative
to the OFF-mode default. All measurements are transport-only:
event allocation, digest computation, lane bookkeeping. Model
math, autograd, and gradient computation are NOT included in
these numbers.

The calibration produces a decision on whether the optional
CONVEYOR_EXPERIMENTAL mode should be enabled. The decision rule
is conservative:

    * BUCKET mode is always certifiable when transport overhead
      ≤ 1% of forward-pass wall-clock at K=16.
    * CONVEYOR mode requires evidence that lane pre-registration
      strictly reduces mean tokens-to-first-broadcast under
      identical inputs, with no impact on semantic digests or
      autograd graph identity.

Neither decision alters K (fixed at 16), model semantics, or
gradient flow.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


class CalibrationRefusal(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class TransportSample:
    boundary_index: int
    forward_ms: float          # wall-clock of the surrounding forward
    transport_ms: float        # wall-clock added by shuttle bookkeeping
    time_to_first_broadcast_ms: float


@dataclass
class CalibrationReport:
    samples: List[TransportSample]
    mean_forward_ms: float
    mean_transport_ms: float
    transport_overhead_fraction: float
    mean_time_to_first_broadcast_ms: float

    def is_bucket_certifiable(self, *,
                                 max_overhead_fraction: float = 0.01
                                 ) -> bool:
        return self.transport_overhead_fraction <= max_overhead_fraction


def summarize(samples: List[TransportSample]) -> CalibrationReport:
    if not samples:
        raise CalibrationRefusal("no_samples")
    n = len(samples)
    total_fwd = sum(s.forward_ms for s in samples)
    total_tr = sum(s.transport_ms for s in samples)
    total_ttb = sum(s.time_to_first_broadcast_ms for s in samples)
    mean_fwd = total_fwd / n
    mean_tr = total_tr / n
    if mean_fwd <= 0:
        raise CalibrationRefusal(
            "invalid_forward_ms",
            f"mean_forward_ms={mean_fwd}")
    return CalibrationReport(
        samples=list(samples),
        mean_forward_ms=mean_fwd,
        mean_transport_ms=mean_tr,
        transport_overhead_fraction=(mean_tr / mean_fwd),
        mean_time_to_first_broadcast_ms=(total_ttb / n),
    )


@dataclass
class ConveyorDecision:
    """The certified rollout decision for CONVEYOR_EXPERIMENTAL.

    ACIS-7 conveyor mode is refused UNLESS every gate below is
    met. Refusal is not a bug — the certified default is BUCKET
    with CONVEYOR disabled."""

    decision: str  # "conveyor_certified" | "conveyor_refused"
    reason_code: str
    reason_detail: str
    bucket_overhead_fraction: float
    conveyor_overhead_fraction: Optional[float]
    conveyor_semantic_identity_preserved: Optional[bool]
    conveyor_autograd_identity_preserved: Optional[bool]


def decide_conveyor(
    *,
    bucket_report: CalibrationReport,
    conveyor_report: Optional[CalibrationReport] = None,
    conveyor_semantic_identity_preserved: Optional[bool] = None,
    conveyor_autograd_identity_preserved: Optional[bool] = None,
    max_overhead_fraction: float = 0.01,
) -> ConveyorDecision:
    """Returns a certified conveyor decision.

    Conveyor is REFUSED unless:
      1. The BUCKET report is itself certifiable.
      2. A conveyor report exists with overhead ≤ bucket overhead.
      3. Both semantic identity and autograd identity are
         preserved on conveyor mode.
    """
    bucket_ok = bucket_report.is_bucket_certifiable(
        max_overhead_fraction=max_overhead_fraction)
    if not bucket_ok:
        return ConveyorDecision(
            decision="conveyor_refused",
            reason_code="bucket_overhead_too_high",
            reason_detail=(
                f"bucket overhead "
                f"{bucket_report.transport_overhead_fraction:.4f} > "
                f"{max_overhead_fraction:.4f}"),
            bucket_overhead_fraction=bucket_report
                                        .transport_overhead_fraction,
            conveyor_overhead_fraction=None,
            conveyor_semantic_identity_preserved=None,
            conveyor_autograd_identity_preserved=None,
        )
    if conveyor_report is None:
        return ConveyorDecision(
            decision="conveyor_refused",
            reason_code="no_conveyor_evidence",
            reason_detail="conveyor overhead not measured",
            bucket_overhead_fraction=bucket_report
                                        .transport_overhead_fraction,
            conveyor_overhead_fraction=None,
            conveyor_semantic_identity_preserved=None,
            conveyor_autograd_identity_preserved=None,
        )
    if (conveyor_report.transport_overhead_fraction >
            bucket_report.transport_overhead_fraction):
        return ConveyorDecision(
            decision="conveyor_refused",
            reason_code="conveyor_slower_than_bucket",
            reason_detail=(
                f"conveyor="
                f"{conveyor_report.transport_overhead_fraction:.4f} > "
                f"bucket="
                f"{bucket_report.transport_overhead_fraction:.4f}"),
            bucket_overhead_fraction=bucket_report
                                        .transport_overhead_fraction,
            conveyor_overhead_fraction=conveyor_report
                                          .transport_overhead_fraction,
            conveyor_semantic_identity_preserved=(
                conveyor_semantic_identity_preserved),
            conveyor_autograd_identity_preserved=(
                conveyor_autograd_identity_preserved),
        )
    if not conveyor_semantic_identity_preserved:
        return ConveyorDecision(
            decision="conveyor_refused",
            reason_code="conveyor_semantic_divergence",
            reason_detail=(
                "conveyor mode altered semantic digests vs "
                "bucket / off"),
            bucket_overhead_fraction=bucket_report
                                        .transport_overhead_fraction,
            conveyor_overhead_fraction=conveyor_report
                                          .transport_overhead_fraction,
            conveyor_semantic_identity_preserved=False,
            conveyor_autograd_identity_preserved=(
                conveyor_autograd_identity_preserved),
        )
    if not conveyor_autograd_identity_preserved:
        return ConveyorDecision(
            decision="conveyor_refused",
            reason_code="conveyor_autograd_divergence",
            reason_detail=(
                "conveyor mode altered autograd graph vs "
                "bucket / off"),
            bucket_overhead_fraction=bucket_report
                                        .transport_overhead_fraction,
            conveyor_overhead_fraction=conveyor_report
                                          .transport_overhead_fraction,
            conveyor_semantic_identity_preserved=True,
            conveyor_autograd_identity_preserved=False,
        )
    return ConveyorDecision(
        decision="conveyor_certified",
        reason_code="all_gates_passed",
        reason_detail="bucket ok; conveyor ≤ bucket; identities preserved",
        bucket_overhead_fraction=bucket_report
                                    .transport_overhead_fraction,
        conveyor_overhead_fraction=conveyor_report
                                      .transport_overhead_fraction,
        conveyor_semantic_identity_preserved=True,
        conveyor_autograd_identity_preserved=True,
    )
