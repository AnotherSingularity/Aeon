"""scripts/en_proof_pilot.py — thin CLI for aeon.en_train.proof_pilot.

Usage:
    python scripts/en_proof_pilot.py --status
    python scripts/en_proof_pilot.py --dry-run   # verifies preconditions only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aeon.en_train.proof_pilot import (
    PilotConfig, halt_state_for_current_environment,
    HALT_AWAITING_DATA, HALT_READY, HALT_FAILED,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", action="store_true",
                    help="print the current halt state and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify preconditions only; do not train")
    args = ap.parse_args()

    cfg = PilotConfig()
    result = halt_state_for_current_environment(cfg)
    print(json.dumps(result, indent=2, sort_keys=True))

    state = result["state"]
    if state == HALT_AWAITING_DATA:
        return 42       # distinct non-zero rc for the halt state
    if state.startswith("READY"):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
