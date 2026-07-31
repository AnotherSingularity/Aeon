"""Mechanical L-series prerequisite lock (W10-0.1).

Consumes the machine-readable W10 closure state at
docs/W10_CLOSURE_STATE.json. Fails if any file matching a
forbidden_pre_closure_l_series_path_pattern exists while any W10 tranche is
not yet closed. This makes it impossible to land L-series runtime code
(reaction-coordinate estimators, barrier registries, intervention harnesses,
Bayes-factor analyses, etc.) before every W10 tranche declares itself
closed via an explicit commit that updates the JSON.

Deliberately narrow:

* Governance / directive text is allowed at any time (see
  `allowed_pre_closure_l_series_paths` in the JSON).
* This lock does not check the CONTENTS of the closed tranches — it only
  checks the CLOSURE FLAG. That flag is a single-line JSON edit and cannot
  be flipped without the diff appearing in the tranche's own commit. The
  reproduction tests in `tests/test_w10_audit_reproduction.py` remain the
  substantive gate on the tranche's actual code changes.
* The JSON is loaded with json.load, not exec'd or eval'd.

If a future W10 tranche needs to add a new L-series-shaped file while
staying inside the gate, either add its path to
`allowed_pre_closure_l_series_paths` (small and reviewed) or wait until the
tranche it belongs to is closed.
"""
import json
import os
import re


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STATE_PATH = os.path.join(ROOT, "docs", "W10_CLOSURE_STATE.json")


def _load_state():
    assert os.path.exists(STATE_PATH), (
        f"W10 closure-state JSON missing at {STATE_PATH!r}. That file is "
        "the L-series prerequisite gate; W10-0.1 must have committed it.")
    with open(STATE_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def _walk_tracked_files():
    """Yield repo-relative POSIX paths. Uses os.walk (test-only) and prunes
    common non-source directories."""
    prune = {".git", "__pycache__", ".build-venv", ".venv", "dist", "build",
              "node_modules", ".mypy_cache", ".pytest_cache"}
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in prune]
        rel_dir = os.path.relpath(dirpath, ROOT).replace(os.sep, "/")
        for fn in filenames:
            yield f"{rel_dir}/{fn}" if rel_dir != "." else fn


# ---------------------------------------------------------------------------
def test_closure_state_json_is_well_formed():
    state = _load_state()
    assert isinstance(state.get("tranches"), dict) and state["tranches"], (
        "closure state must carry a non-empty tranches map")
    for tid, row in state["tranches"].items():
        assert isinstance(row, dict), tid
        assert isinstance(row.get("closed"), bool), (
            f"tranche {tid!r} must carry a boolean `closed` field")
        # The summary is human-readable; require it so nobody adds a bare
        # tranche entry without saying what it does.
        assert row.get("summary"), f"tranche {tid!r} needs a summary"


def test_l_series_gate_declares_allowed_and_forbidden_paths():
    state = _load_state()
    gate = state.get("l_series_gate", {})
    assert isinstance(gate.get("allowed_pre_closure_l_series_paths"), list)
    assert isinstance(gate.get("forbidden_pre_closure_l_series_path_patterns"), list)
    assert gate["forbidden_pre_closure_l_series_path_patterns"], (
        "gate must forbid at least one L-series path pattern; else the "
        "lock is inert")


def test_no_l_series_file_lands_before_every_tranche_is_closed():
    state = _load_state()
    tranches = state["tranches"]
    all_closed = all(row["closed"] for row in tranches.values())
    if all_closed:
        # L-series is unlocked; nothing to enforce here. A companion test
        # (added at W10-11 closure) will start enforcing the L-series own
        # invariants at that point.
        return

    gate = state["l_series_gate"]
    allowed = set(gate.get("allowed_pre_closure_l_series_paths", []))
    forbidden_patterns = [re.compile(p) for p in
                          gate["forbidden_pre_closure_l_series_path_patterns"]]

    offenders = []
    for path in _walk_tracked_files():
        if path in allowed:
            continue
        for pat in forbidden_patterns:
            if pat.search(path):
                offenders.append(path)
                break
    open_tranches = [tid for tid, row in tranches.items() if not row["closed"]]
    assert not offenders, (
        f"L-series file(s) present before W10 closure. Open tranches: "
        f"{open_tranches}. Offending paths: {offenders}. Either close the "
        f"blocking tranche(s) properly (update {os.path.basename(STATE_PATH)}) "
        f"or add the specific path to allowed_pre_closure_l_series_paths.")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
