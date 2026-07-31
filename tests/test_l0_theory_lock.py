"""L0 — theory lock, W10 inheritance, contract stability.

This test is the mechanical guard on L0's stated invariants:

    * W10 Program A closed cleanly and every tranche is `closed: true`
      in docs/W10_CLOSURE_STATE.json.
    * The theory lock document exists and carries the ρ / z definitions,
      the six existence conditions, and the claim ladder.
    * aeon/bypass/contracts.py exposes the frozen L0 contract types
      (WindowTrace, BypassProbe, EvaluationIntervention,
      TensorCaptureBudget, BarrierDefinition, CorpusPartitionManifest).
    * docs/latent_bypass/status.json is well-formed, cross-references
      the theory lock, and records the achieved claim level as 0.
    * The permanent architectural invariants the L-series inherits from
      W10 are still declared where the earlier baselines expect them
      (K=16, Recursion fp32).

No runtime behaviour has changed at L0 — HybridModel.forward must still
be callable without any bypass arguments.
"""
import ast
import json
import os
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

THEORY_LOCK = os.path.join(ROOT, "docs", "LATENT_BYPASS_THEORY_LOCK.md")
STATUS_JSON = os.path.join(ROOT, "docs", "latent_bypass", "status.json")
CONTRACTS = os.path.join(ROOT, "aeon", "bypass", "contracts.py")
W10_CLOSURE = os.path.join(ROOT, "docs", "W10_CLOSURE_STATE.json")


# ---------------------------------------------------------------------------
# W10 inheritance
# ---------------------------------------------------------------------------
def test_w10_program_a_closed_before_l_series_begins():
    with open(W10_CLOSURE, encoding="utf-8") as fh:
        state = json.load(fh)
    open_tranches = [tid for tid, row in state["tranches"].items()
                       if not row["closed"]]
    assert not open_tranches, (
        f"L0 refused: W10 tranches still open: {open_tranches}. Program A "
        "must close before Program B begins.")


def test_l_series_base_commit_recorded():
    from aeon.bypass import L_SERIES_BASE_COMMIT
    assert L_SERIES_BASE_COMMIT, (
        "aeon.bypass must export L_SERIES_BASE_COMMIT so downstream "
        "evidence rows carry the Program-A close commit")
    with open(STATUS_JSON, encoding="utf-8") as fh:
        status = json.load(fh)
    assert status["base_commit"] == L_SERIES_BASE_COMMIT, (
        "docs/latent_bypass/status.json base_commit must match "
        "aeon.bypass.L_SERIES_BASE_COMMIT")


# ---------------------------------------------------------------------------
# Theory lock
# ---------------------------------------------------------------------------
def test_theory_lock_document_exists_and_defines_frame():
    assert os.path.exists(THEORY_LOCK), (
        f"L0 theory lock missing at {THEORY_LOCK!r}")
    src = open(THEORY_LOCK, encoding="utf-8").read()
    # Strip Markdown blockquote markers and collapse whitespace so the
    # phrase matches survive line wrap and `> ` prefixes.
    stripped = "\n".join(l.lstrip("> ").rstrip() for l in src.splitlines())
    collapsed = " ".join(stripped.split())
    assert "visible computational coordinate" in collapsed
    assert "diagnostic projection" in collapsed
    assert "r_b" in src
    assert "K=16" in src or "K = 16" in src
    # The proposition under test
    assert "Aeon uses the hidden state to bypass a barrier" in collapsed
    # The non-metaphysical stance
    assert "not evidence of a bypass" in src or "correlation" in src.lower()
    # The six existence conditions
    for phrase in ("Predictive information", "Causal contribution",
                    "Barrier selectivity", "Net efficiency",
                    "Stability", "Repetition"):
        assert phrase in src, (
            f"L0 theory lock must enumerate the '{phrase}' existence condition")


def test_theory_lock_declares_claim_ladder():
    src = open(THEORY_LOCK, encoding="utf-8").read()
    # The five substantive levels must be named (level 0 is theory-only).
    for name in ("Structurally implemented", "Observational evidence",
                  "Causal-checkpoint", "net-efficiency",
                  "Repeated comparative"):
        assert name.lower() in src.lower(), (
            f"theory lock must name claim level '{name}'")


def test_theory_lock_records_corpus_staging_rule():
    src = open(THEORY_LOCK, encoding="utf-8").read()
    assert "L0" in src and "L3" in src
    assert "synthetic" in src.lower() and "real-English" in src or "real english" in src.lower()
    assert "outbound-network prohibition" in src or "operated on\nentirely offline" in src, (
        "theory lock must record the offline-only operating stance")


# ---------------------------------------------------------------------------
# Contracts module
# ---------------------------------------------------------------------------
def test_contracts_module_exposes_l0_types():
    from aeon.bypass import contracts as c
    for name in ("WindowTrace", "BypassProbe", "EvaluationIntervention",
                  "TensorCaptureBudget", "BarrierDefinition",
                  "CorpusPartitionManifest"):
        assert hasattr(c, name), (
            f"aeon.bypass.contracts must expose {name} at L0")


def test_window_trace_carries_pre_and_post_broadcast_digests():
    """L1 noninterference tests will compare pre-broadcast digests
    across runs; if the L0 contract omits them, L1 can't do its job."""
    from aeon.bypass.contracts import WindowTrace
    tree = ast.parse(open(CONTRACTS, encoding="utf-8").read())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "WindowTrace":
            fields = [n.target.id for n in node.body
                        if isinstance(n, ast.AnnAssign)
                        and isinstance(n.target, ast.Name)]
            for required in ("transformer_pre_broadcast_digest",
                              "transformer_post_broadcast_digest",
                              "recursion_state_before_norm",
                              "recursion_state_after_norm",
                              "recursion_delta_norm",
                              "broadcast_norm",
                              "certificate_margin"):
                assert required in fields, (
                    f"WindowTrace must carry `{required}` field; got {fields}")
            return
    raise AssertionError("WindowTrace class not found in contracts")


def test_bypass_probe_is_a_protocol():
    from aeon.bypass.contracts import BypassProbe
    from typing import Protocol
    # BypassProbe must be a Protocol so multiple probe implementations
    # can satisfy the type contract without a common base class.
    assert Protocol in getattr(BypassProbe, "__mro__", ()), (
        "BypassProbe must be a typing.Protocol")


def test_evaluation_intervention_is_frozen_and_has_kind():
    from aeon.bypass.contracts import EvaluationIntervention
    from dataclasses import fields, is_dataclass
    assert is_dataclass(EvaluationIntervention)
    field_names = {f.name for f in fields(EvaluationIntervention)}
    assert "kind" in field_names, (
        "EvaluationIntervention must carry a 'kind' field so L5 enum "
        "additions don't break the L0 contract")


# ---------------------------------------------------------------------------
# Status JSON
# ---------------------------------------------------------------------------
def test_status_json_well_formed_and_frame_locked():
    with open(STATUS_JSON, encoding="utf-8") as fh:
        s = json.load(fh)
    # L0 achievements and level 0 recorded
    assert s["achieved_claim_level"] == 0, (
        "L0 must record achieved_claim_level=0 — no observation "
        "supports a higher level yet")
    # L0 closed, L1-L11 open
    assert s["tranches"]["L0"]["closed"] is True
    for tid in ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9",
                  "L10", "L11"):
        assert s["tranches"][tid]["closed"] is False, (
            f"tranche {tid} must not be closed at L0")
    # Claim ladder present and ordered
    ladder = s["claim_level_ladder"]
    assert [row["level"] for row in ladder] == [0, 1, 2, 3, 4, 5]
    for row in ladder[2:]:
        assert row["corpus_required"] == "real_english_vendored", (
            f"claim level {row['level']} must require real-English corpus")


def test_status_json_matches_module_claim_levels():
    from aeon.bypass import CLAIM_LEVELS
    with open(STATUS_JSON, encoding="utf-8") as fh:
        s = json.load(fh)
    ladder_names = [row["name"] for row in s["claim_level_ladder"]]
    module_names = [name.split("_", 1)[1] for name in CLAIM_LEVELS]
    assert ladder_names == module_names, (
        f"claim ladder in status.json must match aeon.bypass.CLAIM_LEVELS; "
        f"got status={ladder_names!r} module={module_names!r}")


# ---------------------------------------------------------------------------
# Architectural invariants inherited from W10
# ---------------------------------------------------------------------------
def test_permanent_invariants_still_declared_in_source():
    """The L-series inherits the six V0.02.02 invariants + K=16 + fp32
    Recursion. Verify their declarations remain intact in the source
    (spot-check — the full architecture-preservation suite is the
    authoritative test)."""
    hybrid = open(os.path.join(ROOT, "aeon", "hybrid.py"),
                    encoding="utf-8").read()
    assert "K: int = 16" in hybrid, (
        "hybrid.py must still declare K=16 default at L0 baseline")


def test_default_forward_path_unchanged_by_l0():
    """L0 must not alter HybridModel.forward. This checks the source of
    aeon/hybrid.py for the diagnostic-probe / intervention arguments —
    they must NOT be present until L1 lands (defensive; guards against
    accidental early introduction)."""
    src = open(os.path.join(ROOT, "aeon", "hybrid.py"), encoding="utf-8").read()
    # No bypass import in hybrid.py at L0
    assert "aeon.bypass" not in src, (
        "aeon/hybrid.py must not import aeon.bypass at L0 — the probe "
        "wire-through arrives at L1")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
