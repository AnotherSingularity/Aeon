"""W10-3 — Start / Resume / Recovery are distinct.

Every test drives a real Job through the intent field and the resume-
enumeration helpers, but never opens the Tkinter event loop. That keeps
the tests headless and lets them run on Linux CI alongside the rest of
the suite.
"""
import ast
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from aeon.checkpoint import build_metadata
from aeon.job.key_store import ensure_job_hmac_keyref
from aeon.job.manager import create_job, load_job
from aeon.hybrid import HybridModel
from aeon.transformer import AeonTransformerConfig
from aeon.protected_checkpoint import protected_save, RecoveryDecision
from aeon.launcher.resume import (
    enumerate_checkpoints, latest_authenticated_checkpoint,
    BuildableRecoveryDecision,
)


def _mini_model():
    tcfg = AeonTransformerConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=32, rms_norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=True, attention_bias=False)
    m = HybridModel(
        h_rec=8, K=16, transformer_config=tcfg,
        substrate={"kind": "matrix", "d_in": 8, "d_state": 8,
                    "n_head": 2, "head_size": 4},
        margin_h=0.98, margin_c=0.95,
        freeze_backbone=False, use_embedding_input=True, dtype=torch.float32)
    m.transformer.gamma.data = m.transformer.gamma.data.float()
    return m, tcfg


def _save_authenticated_ckpt(ckpt_dir, step, keyref):
    m, tcfg = _mini_model()
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-3)
    mcfg = {"K": 16, "transformer": tcfg.__dict__}
    md = build_metadata(step=step, model_cfg=mcfg,
                         train_cfg={"lr": 1e-3, "batch_size": 1, "seed": 0},
                         data_cfg={"seq_len": 8},
                         tokenizer_id="sha256:test", corpus_id="sha256:test",
                         data_position=step * 8,
                         instrumentation_cfg={"sample_every": 512, "enabled": True})
    path = os.path.join(ckpt_dir, f"ckpt_{step}.pt")
    protected_save(path, model=m, optimizer=opt, metadata=md,
                    keyref_mac=keyref, authorized_step=step)
    return path


# ---------------------------------------------------------------------------
# Job dataclass carries the three intents
# ---------------------------------------------------------------------------
def _make_job(**kw):
    with tempfile.TemporaryDirectory() as d:
        Path(d).mkdir(exist_ok=True)
        # override jobs_dir to avoid touching LOCALAPPDATA
        os.environ["AEON_DATA_DIR"] = d
        cfg = os.path.join(d, "config.yaml")
        Path(cfg).write_text("model: {}\ntrain: {}\ndata: {}\n", encoding="utf-8")
        job = create_job(
            config_path=cfg,
            tokenizer_path=None, corpus_path=None,
            checkpoint_dir=os.path.join(d, "ck"),
            metrics_dir=os.path.join(d, "m"),
            audit_dir=os.path.join(d, "a"),
            checkpoint_policy={},
            **kw,
        )
    return job


def test_create_job_defaults_intent_to_start():
    with tempfile.TemporaryDirectory() as d:
        os.environ["AEON_DATA_DIR"] = d
        cfg = os.path.join(d, "config.yaml")
        Path(cfg).write_text("", encoding="utf-8")
        job = create_job(
            config_path=cfg,
            tokenizer_path=None, corpus_path=None,
            checkpoint_dir=os.path.join(d, "ck"),
            metrics_dir=os.path.join(d, "m"),
            audit_dir=os.path.join(d, "a"),
            checkpoint_policy={},
        )
        assert job.intent == "start"
        assert job.resume_from_checkpoint is None
        assert job.recovery_decision_path is None


def test_create_job_rejects_unknown_intent():
    with tempfile.TemporaryDirectory() as d:
        os.environ["AEON_DATA_DIR"] = d
        cfg = os.path.join(d, "config.yaml")
        Path(cfg).write_text("", encoding="utf-8")
        try:
            create_job(
                config_path=cfg,
                tokenizer_path=None, corpus_path=None,
                checkpoint_dir=os.path.join(d, "ck"),
                metrics_dir=os.path.join(d, "m"),
                audit_dir=os.path.join(d, "a"),
                checkpoint_policy={},
                intent="rewind",
            )
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_create_job_recover_requires_decision_path():
    with tempfile.TemporaryDirectory() as d:
        os.environ["AEON_DATA_DIR"] = d
        cfg = os.path.join(d, "config.yaml")
        Path(cfg).write_text("", encoding="utf-8")
        try:
            create_job(
                config_path=cfg,
                tokenizer_path=None, corpus_path=None,
                checkpoint_dir=os.path.join(d, "ck"),
                metrics_dir=os.path.join(d, "m"),
                audit_dir=os.path.join(d, "a"),
                checkpoint_policy={},
                intent="recover",
            )
            raise AssertionError("expected ValueError")
        except ValueError:
            pass


def test_load_job_backward_compat_defaults_intent():
    with tempfile.TemporaryDirectory() as d:
        os.environ["AEON_DATA_DIR"] = d
        # Write a pre-W10-3 job.json missing intent / resume_from_checkpoint /
        # recovery_decision_path.
        payload = {
            "job_id": "abc", "job_dir": d, "config_path": "cfg.yaml",
            "tokenizer_path": None, "corpus_path": None,
            "checkpoint_dir": d, "metrics_dir": d, "audit_dir": d,
            "runtime_policy_id": "p", "security_policy_id": "s",
            "checkpoint_policy": {}, "created_at": 0.0,
            "aeon_source_commit": "test", "aeon_release": "test",
        }
        Path(os.path.join(d, "job.json")).write_text(json.dumps(payload))
        job = load_job(d)
        assert job is not None
        assert job.intent == "start"
        assert job.resume_from_checkpoint is None
        assert job.recovery_decision_path is None


# ---------------------------------------------------------------------------
# enumerate_checkpoints + latest_authenticated_checkpoint
# ---------------------------------------------------------------------------
def test_enumerate_returns_authenticated_and_orders_newest_first():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        _save_authenticated_ckpt(ckdir, step=1, keyref=kr)
        _save_authenticated_ckpt(ckdir, step=5, keyref=kr)
        _save_authenticated_ckpt(ckdir, step=3, keyref=kr)
        got = enumerate_checkpoints(ckdir, kr)
        steps = [c.step for c in got if c.authenticated]
        assert steps == [5, 3, 1], steps


def test_enumerate_marks_wrong_key_unauthenticated():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        _save_authenticated_ckpt(ckdir, step=2, keyref=kr)
        # Enumerate with a DIFFERENT key
        with tempfile.TemporaryDirectory() as d2:
            kr2 = ensure_job_hmac_keyref(d2)
            got = enumerate_checkpoints(ckdir, kr2)
            assert len(got) == 1
            assert got[0].authenticated is False
            assert got[0].reason == "MAC mismatch"


def test_latest_authenticated_returns_none_when_empty():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        assert latest_authenticated_checkpoint(ckdir, kr) is None


def test_latest_authenticated_picks_newest():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        _save_authenticated_ckpt(ckdir, step=2, keyref=kr)
        _save_authenticated_ckpt(ckdir, step=9, keyref=kr)
        _save_authenticated_ckpt(ckdir, step=4, keyref=kr)
        cand = latest_authenticated_checkpoint(ckdir, kr)
        assert cand is not None and cand.step == 9


def test_buildable_recovery_decision_shape():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        _save_authenticated_ckpt(ckdir, step=7, keyref=kr)
        cand = latest_authenticated_checkpoint(ckdir, kr)
        brd = BuildableRecoveryDecision(
            candidate=cand,
            reason="corrupted current generation, tested rollback",
            operator_authorization_ref="op:test:20260730",
            current_state_identity="sha256:test-current")
        rd = brd.build()
        assert rd.integrity_result == "verified"
        assert rd.resulting_authorized_state == 7
        js = brd.to_json()
        parsed = json.loads(js)
        assert parsed["reason"] == brd.reason


# ---------------------------------------------------------------------------
# Launcher source structure (A6 flipped: Resume is not aliased to Start)
# ---------------------------------------------------------------------------
def test_launcher_resume_is_not_alias_of_start():
    """W10-3 flipped audit finding A6."""
    src = open(os.path.join(ROOT, "aeon/launcher/gui.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_resume":
            body_source = ast.unparse(node) if hasattr(ast, "unparse") else ""
            # Body must NOT be a single self._on_start() call.
            assert body_source.count("self._on_start()") == 0, (
                "_on_resume must not alias _on_start (W10-3)")
            # Body must reference enumerate/latest_authenticated_checkpoint
            # or intent="resume".
            assert ("latest_authenticated_checkpoint" in body_source
                    or "intent=\"resume\"" in body_source), (
                "_on_resume must enumerate authenticated checkpoints and "
                "spawn with intent='resume'")
            return
    raise AssertionError("_on_resume not found")


def test_launcher_recovery_has_authenticated_flow():
    """W10-3: Recovery is no longer a bare messagebox."""
    src = open(os.path.join(ROOT, "aeon/launcher/gui.py"), encoding="utf-8").read()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_recovery":
            body_source = ast.unparse(node) if hasattr(ast, "unparse") else ""
            assert "intent=\"recover\"" in body_source or "intent='recover'" in body_source, (
                "_on_recovery must spawn a worker with intent='recover'")
            assert "recovery_decision_path" in body_source
            return
    raise AssertionError("_on_recovery not found")


def test_launcher_emits_distinct_audit_event_kinds():
    src = open(os.path.join(ROOT, "aeon/launcher/gui.py"), encoding="utf-8").read()
    for kind in ("start_new_training", "resume_latest", "recovery_authorized"):
        assert kind in src, f"launcher must emit '{kind}' event"


# ---------------------------------------------------------------------------
# Worker consults job.intent, NOT tcfg["resume"]
# ---------------------------------------------------------------------------
def test_worker_reads_job_intent():
    src = open(os.path.join(ROOT, "aeon/job/worker.py"), encoding="utf-8").read()
    # Isolate _run_training_loop.
    import re
    m = re.search(r"def _run_training_loop.*?(?=\n(?:def |\Z))", src, re.DOTALL)
    assert m
    body = m.group(0)
    assert "intent =" in body and "job.intent" in body or 'getattr(job, "intent"' in body, (
        "worker must consult job.intent")
    # Distinct branches per intent
    assert "intent == \"start\"" in body or "intent==\"start\"" in body
    assert "intent in (\"resume\"" in body or "intent == \"resume\"" in body
    assert "recover" in body


def test_worker_start_intent_refuses_existing_chain():
    """W10-3: intent='start' must refuse to overwrite an existing chain."""
    src = open(os.path.join(ROOT, "aeon/job/worker.py"), encoding="utf-8").read()
    assert "start_new_refused_active_chain" in src


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
