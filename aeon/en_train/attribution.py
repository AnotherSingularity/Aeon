"""aeon.en_train.attribution — §22 swap-P2-back attribution test.

Runs the identical evaluation with two weight sets under identical
decoding conditions and confirms the improvement attaches to the
candidate weights.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


@dataclass
class AttributionResult:
    p2_metrics: Dict[str, float]
    candidate_metrics: Dict[str, float]
    p2_restored_metrics: Dict[str, float]
    attribution_confirmed: bool
    reason: str


def _dominant_metric(m: Dict[str, float]) -> float:
    # Use R_readable as the single scalar for a quick attribution
    # decision. Full comparison uses every metric.
    return float(m.get("R_readable", 0.0))


def attribution_test(*, eval_fn: Callable[[str, str], Dict[str, float]],
                          p2_path: str, candidate_path: str
                          ) -> AttributionResult:
    """`eval_fn(load_path, tag) -> metrics_dict` runs the FIXED sealed
    evaluation with the given weights loaded. Called three times:

      1. baseline P2
      2. candidate weights
      3. P2 restored — must return to (1)'s reading
    """
    m_p2_a = eval_fn(p2_path, "baseline_P2")
    m_cand = eval_fn(candidate_path, "candidate")
    m_p2_b = eval_fn(p2_path, "restored_P2")

    r_p2_a = _dominant_metric(m_p2_a)
    r_cand = _dominant_metric(m_cand)
    r_p2_b = _dominant_metric(m_p2_b)

    tol = 1e-6
    restored_matches = abs(r_p2_a - r_p2_b) <= tol
    improved = r_cand > r_p2_a + tol
    if improved and restored_matches:
        return AttributionResult(m_p2_a, m_cand, m_p2_b, True,
                                        "candidate improved over P2 and restoring P2 returned to baseline")
    if improved and not restored_matches:
        return AttributionResult(m_p2_a, m_cand, m_p2_b, False,
                                        "improvement did not vanish on P2 restore — investigate runtime, eval, or prompt path")
    return AttributionResult(m_p2_a, m_cand, m_p2_b, False,
                                    "candidate did not improve over P2 (no attribution to test)")


# ---------------------------------------------------------------------------
# Multi-seed reproduction (§23)
# ---------------------------------------------------------------------------
def summarize_seed_runs(per_seed: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Given {seed -> {passed: bool, ...}}, report all runs and mark
    the aggregate promotion decision (>=2 of 3 must pass)."""
    total = len(per_seed)
    passing = [s for s, r in per_seed.items() if r.get("passed")]
    return {
        "n_seeds": total,
        "n_passing": len(passing),
        "seeds_passing": sorted(passing),
        "aggregate_promotion": (len(passing) >= 2 and total >= 3),
        "per_seed": per_seed,
    }
