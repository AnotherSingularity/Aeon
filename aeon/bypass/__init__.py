"""aeon.bypass — L-series latent-bypass instrumentation and analysis.

This package is written to satisfy the L0 through L11 Latent Bypass
Upgrade directive under the constraints of the theory lock at
docs/LATENT_BYPASS_THEORY_LOCK.md.

Every module in this package is EVALUATION-ONLY. No module here may be
imported by, or alter the behaviour of, `HybridModel.forward` in its
default (probe=None, intervention=None) path. The default forward pass
must be bit-identical whether or not this package is importable.

Module map:

    contracts          — L0 protocols and dataclasses (this tranche).
    signal_trace       — L1 authoritative signal trace (added at L1).
    barriers           — L2 visible-metric barrier registry (added at L2).
    reaction           — L3 hidden reaction-coordinate diagnostics (L3).
    telemetry          — L4 bounded per-window telemetry (L4).
    interventions      — L5 evaluation-only causal interventions (L5).
    inference          — L8 held-out predictive comparison (L8).
    stability          — L9 full-loop stability analysis (L9).

Claim-level bookkeeping lives at docs/latent_bypass/status.json and is
consumed by tests/test_l0_theory_lock.py.
"""
from __future__ import annotations

__all__ = [
    "CLAIM_LEVELS",
    "L_SERIES_BASE_COMMIT",
]

# W10 close (Program A) — the base commit for the L-series work. Used by
# L0 inheritance tests to verify we started from the audited baseline.
L_SERIES_BASE_COMMIT = "7d07a44"

# The claim-level ladder locked at L0. Kept as a tuple so downstream
# code cannot silently reorder or extend it.
CLAIM_LEVELS = (
    "0_THEORY_ONLY",
    "1_STRUCTURALLY_IMPLEMENTED",
    "2_OBSERVATIONAL_EVIDENCE",
    "3_CAUSAL_CHECKPOINT_EVIDENCE",
    "4_SMALL_SCALE_NET_EFFICIENCY_EVIDENCE",
    "5_REPEATED_COMPARATIVE_EVIDENCE",
)
