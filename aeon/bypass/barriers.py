"""L2 visible barrier registry.

Barriers are visible-metric definitions used by the L-series' external
evaluators to identify positions where a purely-visible-state
computation is likely to struggle. Barriers are NOT a new Aeon
routing mechanism. They are external evaluation logic.

Every allowed visible signal (pre-broadcast token loss, pre-broadcast
correct-token margin, target rank, pre-broadcast output entropy,
local-state repetition, task progress, dependency distance, visible
prediction instability, visible failure to resolve a task state)
lives OUTSIDE the model's forward path. No barrier value may enter:

    * the substrate gate
    * Recursion
    * the transformer forward policy
    * training loss weighting
    * token routing
    * adaptive-compute logic
    * any runtime control

The registry loads BarrierDefinition rows from a JSON file (default:
``benchmarks/latent_bypass/barriers.json``) and calibrates thresholds
on a calibration partition. Thresholds LOCK before any evaluation
run touches held-out data — a post-hoc threshold adjustment
invalidates the L-series claim ladder.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# BarrierDefinition — one row in the registry
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BarrierDefinition:
    """One entry in the L2 visible-metric registry.

    ``visible_metric`` must be a name in ALLOWED_VISIBLE_METRICS.
    ``observation_point`` states where in the evaluation pipeline the
    metric is read (pre-broadcast / post-broadcast / task-level).
    ``threshold_method`` names the calibration policy
    (e.g. ``top_percent``, ``fixed_value``, ``per_partition_quantile``).
    ``threshold_value`` is the CALIBRATED threshold — None until the
    calibrator runs on the calibration partition, then locked.
    """

    schema_version: int
    barrier_id: str
    version: int
    description: str
    visible_metric: str
    observation_point: str
    threshold_method: str
    threshold_value: Optional[float]
    calibration_partition: str
    evaluation_partition: str
    minimum_samples: int
    missing_data_action: str
    applicable_tasks: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Allowed visible signals (external evaluator inputs only)
# ---------------------------------------------------------------------------
ALLOWED_VISIBLE_METRICS = frozenset({
    "pre_broadcast_token_loss",
    "pre_broadcast_correct_token_margin",
    "pre_broadcast_target_rank",
    "pre_broadcast_output_entropy",
    "local_state_repetition",
    "task_progress",
    "dependency_distance",
    "visible_prediction_instability",
    "visible_failure_to_resolve_task_state",
})


# Signals the barrier registry MUST NOT be able to reach.
FORBIDDEN_REGISTRY_INPUTS = frozenset({
    "recursion_state",
    "h_cond",
    "broadcast_norm",
    "recursion_delta_norm",
    "h_before",
    "h_after",
    "substrate_state",
    "transformer_attention",
})


class BarrierRegistryError(RuntimeError):
    """Raised on any barrier-registry violation (bad metric name,
    forbidden input, duplicate ID, missing calibration data,
    post-hoc threshold change)."""

    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
class BarrierRegistry:
    """In-memory table of BarrierDefinition rows keyed by barrier_id.

    Constructed via ``load_registry(path)``. Duplicate IDs are refused
    at load time. Forbidden inputs (any field name in
    FORBIDDEN_REGISTRY_INPUTS anywhere in the row) are refused at load
    time.

    ``calibrate(barrier_id, samples)`` sets ``threshold_value`` from
    the calibration samples according to the row's ``threshold_method``
    and returns a new frozen BarrierDefinition. Once calibrated, a
    subsequent call raises unless ``allow_recalibration=True`` is
    passed explicitly (a re-calibration is always visible in the
    audit log the caller supplies).
    """

    def __init__(self, rows: Sequence[BarrierDefinition]):
        by_id: Dict[str, BarrierDefinition] = {}
        for r in rows:
            if r.barrier_id in by_id:
                raise BarrierRegistryError(
                    "duplicate_barrier_id", f"duplicate id: {r.barrier_id!r}")
            if r.visible_metric not in ALLOWED_VISIBLE_METRICS:
                raise BarrierRegistryError(
                    "unknown_visible_metric",
                    f"{r.barrier_id}: {r.visible_metric!r} not in "
                    "ALLOWED_VISIBLE_METRICS")
            by_id[r.barrier_id] = r
        self._rows: Dict[str, BarrierDefinition] = by_id

    def __contains__(self, barrier_id: str) -> bool:
        return barrier_id in self._rows

    def __len__(self) -> int:
        return len(self._rows)

    def ids(self) -> List[str]:
        return sorted(self._rows.keys())

    def get(self, barrier_id: str) -> BarrierDefinition:
        try:
            return self._rows[barrier_id]
        except KeyError as e:
            raise BarrierRegistryError(
                "unknown_barrier_id", f"{barrier_id!r}") from e

    def is_calibrated(self, barrier_id: str) -> bool:
        return self._rows[barrier_id].threshold_value is not None

    def calibrate(
        self,
        barrier_id: str,
        samples: Sequence[float],
        *,
        allow_recalibration: bool = False,
    ) -> BarrierDefinition:
        row = self.get(barrier_id)
        if row.threshold_value is not None and not allow_recalibration:
            raise BarrierRegistryError(
                "already_calibrated",
                f"{barrier_id}: threshold already locked")
        if len(samples) < row.minimum_samples:
            raise BarrierRegistryError(
                "insufficient_calibration_samples",
                f"{barrier_id}: got {len(samples)} < required "
                f"{row.minimum_samples}")
        method = row.threshold_method
        if method == "top_percent":
            # threshold = value such that top X% of the calibration
            # distribution exceeds it. X is encoded in the id suffix
            # for now (documented per definition).
            xs = sorted(samples)
            # Default to top 5%.
            k = max(1, int(0.05 * len(xs)))
            threshold = xs[-k]
        elif method == "bottom_percent":
            xs = sorted(samples)
            k = max(1, int(0.05 * len(xs)))
            threshold = xs[k - 1]
        elif method == "median":
            xs = sorted(samples)
            threshold = xs[len(xs) // 2]
        elif method == "fixed_value":
            # threshold_value must have been supplied at load time.
            raise BarrierRegistryError(
                "fixed_value_not_calibratable",
                f"{barrier_id}: fixed_value threshold must be set in "
                "the registry file, not calibrated at runtime")
        elif method == "per_partition_quantile":
            xs = sorted(samples)
            k = max(1, int(0.9 * len(xs))) - 1
            threshold = xs[k]
        else:
            raise BarrierRegistryError(
                "unknown_threshold_method",
                f"{barrier_id}: {method!r}")
        new_row = BarrierDefinition(
            schema_version=row.schema_version,
            barrier_id=row.barrier_id,
            version=row.version,
            description=row.description,
            visible_metric=row.visible_metric,
            observation_point=row.observation_point,
            threshold_method=row.threshold_method,
            threshold_value=float(threshold),
            calibration_partition=row.calibration_partition,
            evaluation_partition=row.evaluation_partition,
            minimum_samples=row.minimum_samples,
            missing_data_action=row.missing_data_action,
            applicable_tasks=row.applicable_tasks,
        )
        self._rows[barrier_id] = new_row
        return new_row

    def evaluate(
        self,
        barrier_id: str,
        metric_value: Optional[float],
    ) -> bool:
        """Return True if the metric value exceeds the row's threshold
        (top_percent / median methods) or falls below (bottom_percent).
        ``missing_data_action`` dictates the response when
        ``metric_value`` is None."""
        row = self.get(barrier_id)
        if row.threshold_value is None:
            raise BarrierRegistryError(
                "not_calibrated",
                f"{barrier_id}: threshold unset — call calibrate() first")
        if metric_value is None:
            action = row.missing_data_action
            if action == "false":
                return False
            if action == "true":
                return True
            if action == "raise":
                raise BarrierRegistryError(
                    "missing_metric_value",
                    f"{barrier_id}: metric value is None")
            raise BarrierRegistryError(
                "unknown_missing_data_action",
                f"{barrier_id}: {action!r}")
        method = row.threshold_method
        if method in ("top_percent", "per_partition_quantile", "median"):
            return float(metric_value) >= row.threshold_value
        if method == "bottom_percent":
            return float(metric_value) <= row.threshold_value
        if method == "fixed_value":
            return float(metric_value) >= row.threshold_value
        raise BarrierRegistryError(
            "unknown_threshold_method", f"{barrier_id}: {method!r}")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_registry(path: str) -> BarrierRegistry:
    """Load a barrier-registry JSON file and construct a
    BarrierRegistry. Refuses on schema violations."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    if raw.get("schema_version") != 1:
        raise BarrierRegistryError(
            "unsupported_schema_version",
            f"{path}: schema_version={raw.get('schema_version')!r}")
    rows_raw = raw.get("barriers", [])
    if not rows_raw:
        raise BarrierRegistryError(
            "empty_registry", f"{path}: no barrier rows")
    rows = []
    for r in rows_raw:
        _reject_forbidden_inputs(r)
        rows.append(BarrierDefinition(
            schema_version=int(r.get("schema_version", 1)),
            barrier_id=r["barrier_id"],
            version=int(r["version"]),
            description=r["description"],
            visible_metric=r["visible_metric"],
            observation_point=r["observation_point"],
            threshold_method=r["threshold_method"],
            threshold_value=(float(r["threshold_value"])
                              if r.get("threshold_value") is not None else None),
            calibration_partition=r["calibration_partition"],
            evaluation_partition=r["evaluation_partition"],
            minimum_samples=int(r["minimum_samples"]),
            missing_data_action=r["missing_data_action"],
            applicable_tasks=tuple(r.get("applicable_tasks", ())),
        ))
    return BarrierRegistry(rows)


def _reject_forbidden_inputs(raw: Mapping[str, Any]) -> None:
    """Static defensive check: no field name in the row may reference
    a hidden-state input. This guards against a registry file being
    edited to sneak Recursion state into the barrier metric."""
    def _walk(obj):
        if isinstance(obj, str):
            if obj in FORBIDDEN_REGISTRY_INPUTS:
                raise BarrierRegistryError(
                    "forbidden_hidden_state_input",
                    f"barrier row references {obj!r}")
        elif isinstance(obj, Mapping):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)
    _walk(raw)


# ---------------------------------------------------------------------------
# Registry file digest (for evidence)
# ---------------------------------------------------------------------------
def registry_digest(path: str) -> str:
    """SHA-256 of the registry file bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"
