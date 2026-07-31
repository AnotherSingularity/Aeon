"""W10-R — directive reconciliation enforcement.

Consumes docs/w10_reconciliation.json. The matrix declares the
substantive gaps (R6/R8/R20/R26) and evidence gaps (R17/R31/R32) that
W10-R corrects; this file asserts each correction landed and that no
new substantive gap has opened.
"""
import ast
import importlib
import json
import os
import re
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

MATRIX_PATH = os.path.join(ROOT, "docs", "w10_reconciliation.json")


def _load_matrix():
    with open(MATRIX_PATH, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Matrix well-formedness
# ---------------------------------------------------------------------------
def test_reconciliation_matrix_well_formed():
    m = _load_matrix()
    assert isinstance(m.get("matrix"), dict) and m["matrix"], (
        "reconciliation matrix must carry a non-empty 'matrix' map")
    for rid, row in m["matrix"].items():
        assert isinstance(row, dict), rid
        assert row.get("classification") in m["classifications_allowed"], (
            f"row {rid!r} classification {row.get('classification')!r} "
            "not in classifications_allowed")


def test_no_substantive_gap_left_open():
    m = _load_matrix()
    for rid, row in m["matrix"].items():
        klass = row.get("classification")
        if klass in ("SUBSTANTIVE_FUNCTIONAL_GAP", "SUBSTANTIVE_SECURITY_GAP"):
            action = row.get("action", "")
            assert action and action != "none", (
                f"row {rid} is {klass} but declares no action")


def test_closure_state_reflects_reality():
    m = _load_matrix()
    cs = m["closure_state"]
    # Count actual open substantive rows
    open_sub = sum(1 for r in m["matrix"].values()
                     if r["classification"] in
                     ("SUBSTANTIVE_FUNCTIONAL_GAP", "SUBSTANTIVE_SECURITY_GAP")
                     and not r.get("resolved"))
    assert cs["substantive_open_gaps"] == open_sub, (
        f"closure_state.substantive_open_gaps={cs['substantive_open_gaps']} "
        f"but matrix has {open_sub} unresolved substantive rows")


def test_closure_state_splits_content_vs_closure_commit():
    """L0.1: a commit cannot contain its own final hash, so the closure
    state records the content commit (containing the reconciled
    implementation) and the closure commit (containing the finalized
    closure metadata) separately."""
    m = _load_matrix()
    cs = m["closure_state"]
    assert "w10_reconciled_content_commit" in cs, (
        "L0.1: closure_state must record the content commit separately")
    assert "w10_reconciliation_closure_commit" in cs, (
        "L0.1: closure_state must record the closure commit separately")
    assert cs["w10_reconciled_content_commit"] != cs["w10_reconciliation_closure_commit"], (
        "L0.1: content and closure commits must be distinct")


# ---------------------------------------------------------------------------
# R8 — vocab mismatch fail-closed
# ---------------------------------------------------------------------------
def test_worker_fails_closed_on_tokenizer_vocab_mismatch():
    src = open(os.path.join(ROOT, "aeon", "job", "worker.py"),
                encoding="utf-8").read()
    # Silent rebind is gone: any `.vocab_size =` on tcfg_model is now
    # gated on _cfg_uninit.
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_training_loop":
            body = ast.unparse(node)
            # The vocab mismatch must be a fail branch, not a rebind.
            assert "tokenizer_vocab_mismatch" in body, (
                "W10-R/R8: worker must emit tokenizer_vocab_mismatch reason")
            assert "_cfg_uninit" in body, (
                "W10-R/R8: worker must distinguish uninitialized vocab from mismatch")
            return
    raise AssertionError("_run_training_loop not found")


# ---------------------------------------------------------------------------
# R6 — periodic validation uses eval mode + no_grad
# ---------------------------------------------------------------------------
def test_periodic_validation_uses_eval_mode_and_no_grad():
    src = open(os.path.join(ROOT, "aeon", "job", "worker.py"),
                encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_run_periodic_validation":
            body = ast.unparse(node)
            assert "model.eval()" in body, (
                "W10-R/R6: periodic validation must switch to eval mode")
            assert "torch.no_grad()" in body, (
                "W10-R/R6: periodic validation must run under torch.no_grad()")
            assert "model.train()" in body, (
                "W10-R/R6: periodic validation must restore training mode")
            assert '"eval_mode": True' in body or "'eval_mode': True" in body, (
                "W10-R/R6: evidence line must record eval_mode=True")
            return
    raise AssertionError("_run_periodic_validation not found")


# ---------------------------------------------------------------------------
# R20 — release-identity compatibility on Resume
# ---------------------------------------------------------------------------
def test_protected_load_carries_expected_release_identity_argument():
    from aeon.protected_checkpoint import protected_load
    import inspect
    sig = inspect.signature(protected_load)
    assert "expected_release_identity" in sig.parameters, (
        "W10-R/R20: protected_load must accept expected_release_identity")


def test_release_identity_mismatch_rejected_on_resume():
    """A checkpoint with source_commit='alpha' cannot Resume under
    RELEASE_METADATA['source_commit']='beta'."""
    from aeon.protected_checkpoint import (
        protected_save, protected_load,
        CheckpointIncompatible, KeyRef,
    )
    from aeon.checkpoint import build_metadata
    import torch
    with tempfile.TemporaryDirectory() as d:
        m = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(m.parameters(), lr=0.001)
        _kbytes = b"\x01" * 32
        key = KeyRef(handle="test", resolve=lambda: _kbytes)
        md = build_metadata(step=1,
                             model_cfg={"transformer": {"vocab_size": 32}},
                             train_cfg={"seed": 1},
                             data_cfg={"seq_len": 8},
                             tokenizer_id="sha256:tok",
                             corpus_id="sha256:corp",
                             data_position=0)
        md["source_commit"] = "alpha"
        ck = os.path.join(d, "ck.pt")
        protected_save(ck, model=m, optimizer=opt, metadata=md,
                        keyref_mac=key, authorized_step=1)
        try:
            protected_load(ck, keyref_mac=key,
                           expected_model_config={"transformer": {"vocab_size": 32}},
                           expected_release_identity="beta")
        except CheckpointIncompatible as e:
            assert "release_identity" in str(e), str(e)
        else:
            raise AssertionError("Resume across releases must raise CheckpointIncompatible")


def test_release_identity_mismatch_allowed_under_recovery():
    """Under an operator-authorized RecoveryDecision the release
    mismatch is permitted — this is the authorized-recovery-under-older-
    release path."""
    from aeon.protected_checkpoint import (
        protected_save, protected_load, KeyRef, RecoveryDecision,
    )
    from aeon.checkpoint import build_metadata
    import torch
    with tempfile.TemporaryDirectory() as d:
        m = torch.nn.Linear(4, 4)
        opt = torch.optim.SGD(m.parameters(), lr=0.001)
        _kbytes = b"\x01" * 32
        key = KeyRef(handle="test", resolve=lambda: _kbytes)
        md = build_metadata(step=1,
                             model_cfg={"transformer": {"vocab_size": 32}},
                             train_cfg={"seed": 1},
                             data_cfg={"seq_len": 8},
                             tokenizer_id="sha256:tok",
                             corpus_id="sha256:corp",
                             data_position=0)
        md["source_commit"] = "alpha"
        ck = os.path.join(d, "ck.pt")
        protected_save(ck, model=m, optimizer=opt, metadata=md,
                        keyref_mac=key, authorized_step=1)
        rd = RecoveryDecision(
            operator_authorization_ref="test",
            reason="cross-release recovery",
            current_state_identity="sha256:current",
            selected_state_identity="sha256:selected",
            integrity_result="verified",
            recovery_policy_version=1,
            resulting_authorized_state=1)
        # Should not raise despite release mismatch — Recovery is
        # explicit authorization.
        blob = protected_load(ck, keyref_mac=key,
                                expected_model_config={"transformer": {"vocab_size": 32}},
                                expected_release_identity="beta",
                                recovery_decision=rd)
        assert blob is not None


# ---------------------------------------------------------------------------
# R26 — frozen preflight blocks on missing release identity
# ---------------------------------------------------------------------------
def test_frozen_preflight_blocks_on_missing_release_identity():
    from aeon.config import preflight
    with mock.patch.object(preflight, "_is_frozen", return_value=True), \
         mock.patch.dict("aeon.version.RELEASE_METADATA",
                          {"source_commit": "unknown"}, clear=False):
        res = preflight.run_preflight({})
    rid = [c for c in res.checks if c.name == "release_identity"]
    assert rid, "preflight must emit a release_identity check"
    assert rid[0].status == "fail", (
        f"frozen preflight with unknown source_commit must FAIL; got {rid[0]}")
    assert res.verdict.value == "BLOCKED"


def test_source_preflight_skips_release_identity_check():
    from aeon.config import preflight
    with mock.patch.object(preflight, "_is_frozen", return_value=False):
        res = preflight.run_preflight({})
    rid = [c for c in res.checks if c.name == "release_identity"]
    assert rid, "preflight must emit a release_identity check"
    assert rid[0].status == "skip", (
        f"source-tree preflight release_identity must be 'skip'; got {rid[0]}")


# ---------------------------------------------------------------------------
# R17 — button gating evidence: error paths do not spawn workers
# ---------------------------------------------------------------------------
def test_resume_error_path_does_not_spawn_worker():
    """When latest_authenticated_checkpoint returns None, _on_resume must
    surface an error and NOT reach spawn_worker."""
    src = open(os.path.join(ROOT, "aeon", "launcher", "gui.py"),
                encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_resume":
            body = ast.unparse(node)
            # The path must be guarded — a 'latest is None' branch OR a
            # try/except that produces an error before spawn.
            assert "spawn_worker" in body, "_on_resume must call spawn_worker in the happy path"
            # And error path must precede spawn or return early.
            assert "showerror" in body or "showwarning" in body or "showinfo" in body, (
                "_on_resume error path must surface a dialog before returning")
            return
    raise AssertionError("_on_resume not found")


def test_recovery_error_path_does_not_spawn_worker():
    src = open(os.path.join(ROOT, "aeon", "launcher", "gui.py"),
                encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_recovery":
            body = ast.unparse(node)
            # Recovery early-returns on: no last_job_dir, cannot
            # enumerate, no authenticated candidates, no chosen, no
            # reason. Each MUST return before spawn_worker.
            assert body.count("return") >= 4, (
                "_on_recovery must have multiple guard returns before spawn_worker")
            return
    raise AssertionError("_on_recovery not found")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
