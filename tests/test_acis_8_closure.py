"""ACIS-8 — integrated closure invariants.

Locks the certified return state STATE_A. Every check here is an
enforcement gate for a claim in the certification report.
"""
import ast
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Structural closure: docs + evidence exist and are consistent
# ---------------------------------------------------------------------------
def test_closure_documents_present():
    for rel in ("docs/acis/ACIS_ARCHITECTURE.md",
                    "docs/acis/ACIS_BASELINE_REPORT.md",
                    "docs/acis/ACIS_CERTIFICATION_REPORT.md",
                    "docs/acis/acis_final_evidence.json",
                    "docs/acis/acis_status.json"):
        p = os.path.join(ROOT, rel)
        assert os.path.exists(p), f"missing closure artifact: {rel}"
        assert os.path.getsize(p) > 200, f"suspiciously small: {rel}"


def test_evidence_json_declares_state_A():
    p = os.path.join(ROOT, "docs", "acis", "acis_final_evidence.json")
    with open(p, encoding="utf-8") as f:
        ev = json.load(f)
    assert ev["return_state"] == "STATE_A_ACIS_COMPLETE"
    assert ev["regression"]["failures"] == 0


def test_status_json_marks_every_tranche_closed():
    p = os.path.join(ROOT, "docs", "acis", "acis_status.json")
    with open(p, encoding="utf-8") as f:
        st = json.load(f)
    for k in ("ACIS-0", "ACIS-1", "ACIS-2", "ACIS-3", "ACIS-4",
                "ACIS-5", "ACIS-6", "ACIS-7", "ACIS-8"):
        assert st["tranches"][k]["closed"] is True, (
            f"tranche {k} still marked open in acis_status.json")


def test_status_json_locks_conveyor_default_refused():
    p = os.path.join(ROOT, "docs", "acis", "acis_status.json")
    with open(p, encoding="utf-8") as f:
        st = json.load(f)
    assert st["conveyor_certified"] is False
    assert st["mode_default"] == "off"


# ---------------------------------------------------------------------------
# Invariant sweeps
# ---------------------------------------------------------------------------
def test_K_is_fixed_at_16_everywhere():
    from aeon.shuttle import FIXED_K
    from aeon.hybrid import HybridModel  # noqa: F401
    assert FIXED_K == 16
    src = open(os.path.join(ROOT, "aeon", "hybrid.py"),
                 encoding="utf-8").read()
    assert "K: int = 16" in src or "K = 16" in src


def test_no_AdaptiveKControlCapsule_anywhere():
    for dirpath, _, files in os.walk(os.path.join(ROOT, "aeon")):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            with open(p, encoding="utf-8", errors="ignore") as f:
                assert "AdaptiveKControlCapsule" not in f.read(), p


def test_no_executed_recursion_iterations_assertion_in_tests():
    forbidden_token = "executed" "_recursion_" "iterations"  # split so this file is not a false positive
    for dirpath, _, files in os.walk(os.path.join(ROOT, "tests")):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            if fn == "test_acis_8_closure.py":
                continue
            p = os.path.join(dirpath, fn)
            with open(p, encoding="utf-8", errors="ignore") as f:
                src = f.read()
            assert forbidden_token not in src, p


def test_shuttle_package_has_no_outbound_network():
    forbidden = ("urllib.request", "requests.get", "requests.post",
                    "httpx.get", "httpx.post", "socket.socket(",
                    "urlopen(")
    for dirpath, _, files in os.walk(os.path.join(ROOT, "aeon",
                                                        "shuttle")):
        for fn in files:
            if not fn.endswith(".py"):
                continue
            p = os.path.join(dirpath, fn)
            with open(p, encoding="utf-8", errors="ignore") as f:
                src = f.read()
            for tok in forbidden:
                assert tok not in src, f"{p}: {tok}"


def test_hybrid_forward_shuttle_guard_present():
    src = open(os.path.join(ROOT, "aeon", "hybrid.py"),
                 encoding="utf-8").read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            else:
                names = [node.module or ""]
            for n in names:
                assert not n.startswith("aeon.shuttle"), (
                    f"module-level import of {n} — shuttle must be optional")
    assert "if shuttle is not None:" in src


# ---------------------------------------------------------------------------
# Ledger cross-check: commit ledger in evidence json enumerates 9 tranches
# ---------------------------------------------------------------------------
def test_evidence_ledger_lists_all_nine_tranches():
    p = os.path.join(ROOT, "docs", "acis", "acis_final_evidence.json")
    with open(p, encoding="utf-8") as f:
        ev = json.load(f)
    ledger = ev["commit_ledger"]
    for k in ("ACIS-0", "ACIS-1", "ACIS-2", "ACIS-3", "ACIS-4",
                "ACIS-5", "ACIS-6", "ACIS-7", "ACIS-8"):
        assert k in ledger, k


# ---------------------------------------------------------------------------
# Prior ACIS test files exist
# ---------------------------------------------------------------------------
def test_all_prior_acis_test_files_exist():
    for rel in (
        "tests/test_acis_0_baseline.py",
        "tests/test_acis_1_broadcast.py",
        "tests/test_acis_2_leases.py",
        "tests/test_acis_3_shuttle.py",
        "tests/test_acis_4_ownership.py",
        "tests/test_acis_5_lane.py",
        "tests/test_acis_6_recovery.py",
        "tests/test_acis_7_calibration.py",
    ):
        assert os.path.exists(os.path.join(ROOT, rel)), rel


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
