"""aeon.shuttle — ACIS Coherent Information Shuttling.

A transport and lifecycle layer BENEATH Aeon's cognition. Never
changes:

    * Transformer / substrate / Recursion semantics.
    * Recursion update timing (K=16 slow-clock).
    * Model parameters, training objective, tokenizer, corpus path.
    * Substrate gate inputs.
    * Stream isolation (no direct transformer↔substrate call).
    * Single-broadcast semantics.

Default mode is OFF. In OFF mode, HybridModel.forward runs unchanged;
no ACIS code executes and no allocation, event, or lease is produced.

Rollout modes live in ``aeon.shuttle.policy.ShuttleMode``. Every
tranche in the ACIS commit ledger lands additively; nothing in this
package is imported by ``HybridModel.forward``'s default path.
"""
from __future__ import annotations

__all__ = [
    "FIXED_K",
    "SHUTTLE_MODE_DEFAULT",
    "BASE_COMMIT_ACIS_0",
]

# Fixed slow-clock interval. IMMUTABLE per certified architecture.
FIXED_K: int = 16

# Baseline commit — the ACIS_0 tranche starts here. Referenced in
# docs/acis/acis_status.json.
BASE_COMMIT_ACIS_0: str = "ee28f48"

# The default mode. See aeon.shuttle.policy.ShuttleMode.
SHUTTLE_MODE_DEFAULT: str = "off"
