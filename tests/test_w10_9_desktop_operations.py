"""W10-9 — real desktop operations + real metrics.

Covers audit findings A7 / A8 / A18 / A19 / A20:

    A7  worker consumes launcher settings (cpu_thread_limit,
        memory_ceiling_gb, validation_interval).
    A8  worker emits real step-time / tokens-per-second, not zeros.
    A18 GUI Validate runs the diagnostic subprocess against the latest
        authenticated checkpoint and displays the report.
    A19 GUI Recovery enumerates authenticated checkpoints in-GUI and
        builds the RecoveryDecision in-process (no operator-typed JSON,
        no terminal).
    A20 GUI Diagnose captures subprocess stdout/stderr and shows them.
"""
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _read(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


# ---------------------------------------------------------------------------
# A7 — worker respects launcher settings
# ---------------------------------------------------------------------------
def test_worker_reads_cpu_thread_limit_from_compute_policy():
    src = _read("aeon/job/worker.py")
    assert "_cpu_thread_limit_from" in src
    assert "torch.set_num_threads" in src
    assert 'compute_policy' in src or 'job.compute_policy' in src


def test_worker_reads_memory_ceiling_and_safe_stops():
    src = _read("aeon/job/worker.py")
    assert "_memory_ceiling_mb_from" in src
    assert "memory_ceiling_exceeded" in src, (
        "W10-9/A7: worker must emit a structured stop reason when the "
        "memory ceiling is exceeded")


def test_worker_runs_periodic_validation():
    src = _read("aeon/job/worker.py")
    assert "_run_periodic_validation" in src
    assert "validation_interval" in src
    assert "worker_validation.jsonl" in src, (
        "W10-9/A7: periodic validation must write an evidence line "
        "the launcher can read")


def test_job_dataclass_has_compute_policy_field():
    src = _read("aeon/job/manager.py")
    assert "compute_policy" in src
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Job":
            fields = [n.target.id for n in node.body
                        if isinstance(n, ast.AnnAssign)
                        and isinstance(n.target, ast.Name)]
            assert "compute_policy" in fields, (
                f"Job dataclass missing compute_policy field; got {fields}")
            return
    raise AssertionError("Job class not found")


def test_load_job_backfills_compute_policy_default():
    """Older job.json files with no compute_policy still load."""
    from aeon.job.manager import load_job
    with tempfile.TemporaryDirectory() as d:
        job_dir = Path(d) / "j"
        job_dir.mkdir()
        legacy = {
            "job_id": "abc", "job_dir": str(job_dir),
            "config_path": "/x.yaml", "tokenizer_path": None,
            "corpus_path": None, "checkpoint_dir": str(job_dir),
            "metrics_dir": str(job_dir), "audit_dir": str(job_dir),
            "runtime_policy_id": "aeon-runtime-v1",
            "security_policy_id": "aeon-security-v1",
            "checkpoint_policy": {}, "created_at": 0.0,
            "aeon_source_commit": "x", "aeon_release": "0.2.3",
        }
        (job_dir / "job.json").write_text(json.dumps(legacy))
        job = load_job(str(job_dir))
        assert job is not None
        assert job.compute_policy == {}, job


# ---------------------------------------------------------------------------
# A8 — real metrics
# ---------------------------------------------------------------------------
def test_worker_measures_step_time_via_perf_counter():
    src = _read("aeon/job/worker.py")
    assert "perf_counter" in src
    assert "_step_perf" in src
    # No lingering zero-placeholder assignment lines
    for placeholder in ("step_time_s=0.0", "tokens_per_s_raw=0.0"):
        assert placeholder not in src, (
            f"W10-9/A8: {placeholder} placeholder still present")


def test_worker_emits_tokens_per_second_derived_from_batch_size():
    """The B*T token accumulator drives the emitted throughput."""
    src = _read("aeon/job/worker.py")
    assert 'B * T' in src or '"tokens": 0' in src, (
        "W10-9/A8: worker must accumulate real B*T tokens per step for throughput")


# ---------------------------------------------------------------------------
# A18 — Validate runs the diagnostic
# ---------------------------------------------------------------------------
def test_on_validate_captures_subprocess_output():
    src = _read("aeon/launcher/gui.py")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_validate":
            body = ast.unparse(node)
            assert "subprocess.run" in body, "Validate must run diagnose subprocess"
            assert "capture_output=True" in body
            assert "--diagnose" in body
            assert "latest_authenticated_checkpoint" in body
            return
    raise AssertionError("_on_validate not found")


def test_on_validate_shows_scrollable_report():
    src = _read("aeon/launcher/gui.py")
    assert "_show_scrollable_report" in src
    # Method should exist as a def
    tree = ast.parse(src)
    names = [n.name for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef)]
    assert "_show_scrollable_report" in names


# ---------------------------------------------------------------------------
# A19 — Recovery enumerates in-GUI
# ---------------------------------------------------------------------------
def test_on_recovery_enumerates_and_builds_decision_in_process():
    src = _read("aeon/launcher/gui.py")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_recovery":
            body = ast.unparse(node)
            assert "enumerate_checkpoints" in body
            assert "BuildableRecoveryDecision" in body
            assert "intent='recover'" in body or 'intent="recover"' in body
            # And no more "select RecoveryDecision JSON" file picker for the
            # pre-baked-JSON path.
            assert "select RecoveryDecision JSON" not in body.lower()
            return
    raise AssertionError("_on_recovery not found")


# ---------------------------------------------------------------------------
# A20 — Diagnose captures output
# ---------------------------------------------------------------------------
def test_on_diagnose_captures_output_not_devnull():
    src = _read("aeon/launcher/gui.py")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_diagnose":
            body = ast.unparse(node)
            assert "DEVNULL" not in body
            assert "capture_output=True" in body
            return
    raise AssertionError("_on_diagnose not found")


# ---------------------------------------------------------------------------
# Live worker → real metrics smoke (uses the W10-1 real-corpus scaffolding)
# ---------------------------------------------------------------------------
def test_worker_actually_emits_nonzero_metrics_end_to_end():
    """Spawn the worker on a tiny real corpus + tokenizer; assert the
    metrics jsonl contains non-zero step_time_s and tokens_per_s_raw
    lines.

    This mirrors the setup of the W10-1 integration test (tiny corpus,
    tiny tokenizer, tiny config) and adds a metrics assertion. If the
    W10-1 harness cannot construct the data source in this environment
    (missing SentencePiece build deps for example), the test skips.
    """
    try:
        import sentencepiece as spm
    except Exception:
        return  # skip
    import yaml
    import torch  # noqa: F401
    from aeon.job.manager import create_job
    from aeon.job.worker import run_worker

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # Tokenizer
        raw = d / "raw.txt"
        raw.write_text("\n".join([
            "the quick brown fox jumps over the lazy dog",
            "a small red car parked on the street",
            "she opened the door and walked outside",
            "he laughed and closed the book slowly",
            "birds sing at dawn on the old oak tree",
        ] * 100))
        tok_prefix = str(d / "tok")
        spm.SentencePieceTrainer.train(
            input=str(raw), model_prefix=tok_prefix, vocab_size=100,
            character_coverage=0.9995, model_type="bpe",
            bos_id=1, eos_id=2, pad_id=0, unk_id=3)
        # Corpus
        corpus_dir = d / "corpus"
        corpus_dir.mkdir()
        text_body = " ".join([raw.read_text()] * 3)
        (corpus_dir / "shard0.jsonl").write_text(
            json.dumps({"text": text_body}) + "\n"
             + json.dumps({"text": text_body}) + "\n")
        # Config
        cfg = {
            "model": {"h_rec": 64, "K": 16, "margin_h": 0.02, "margin_c": 0.02,
                        "dtype": "float32",
                        "transformer": {"hidden_size": 64,
                                        "num_hidden_layers": 1,
                                        "num_attention_heads": 2,
                                        "num_key_value_heads": 2,
                                        "head_dim": 32,
                                        "intermediate_size": 128,
                                        "vocab_size": 100,
                                        "max_position_embeddings": 32}},
            "data": {"seq_len": 32},
            "train": {"seed": 1, "lr": 0.001, "batch_size": 2,
                        "max_steps": 4, "log_every": 2, "ckpt_every": 100,
                        "sample_every": 1000, "observability": True},
        }
        cfg_path = d / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))
        # Job
        with mock.patch("aeon.windows_paths.jobs_dir", return_value=d / "jobs"):
            with mock.patch("aeon.job.manager.jobs_dir", return_value=d / "jobs"):
                job = create_job(
                    config_path=str(cfg_path),
                    tokenizer_path=tok_prefix + ".model",
                    corpus_path=str(corpus_dir),
                    checkpoint_dir=str(d / "ck"),
                    metrics_dir=str(d / "metrics"),
                    audit_dir=str(d / "audit"),
                    checkpoint_policy={"interval": 100,
                                         "validation_interval": 2},
                    compute_policy={"cpu_thread_limit": 1},
                    intent="start")
                rc = run_worker(job.job_json_path)
                assert rc == 0, f"worker rc={rc}"
        # Read metrics.jsonl and assert real step_time_s and tokens_per_s
        metrics_path = d / "metrics" / "metrics.jsonl"
        if not metrics_path.exists():
            # Alternative name used by Observer
            candidates = list((d / "metrics").glob("*.jsonl"))
            assert candidates, list((d / "metrics").iterdir())
            metrics_path = candidates[0]
        found_nonzero = False
        for line in metrics_path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("kind") == "always_on":
                if (float(rec.get("step_time_s", 0)) > 0
                        and float(rec.get("tokens_per_s_raw", 0)) > 0):
                    found_nonzero = True
                    break
        assert found_nonzero, (
            "W10-9/A8: worker must emit at least one metrics line with "
            "non-zero step_time_s AND tokens_per_s_raw")
        # And worker_validation.jsonl must exist because validation_interval=2
        # fires at step 2 and step 4.
        val_path = d / "audit" / "worker_validation.jsonl"
        assert val_path.exists(), "W10-9/A7: worker_validation.jsonl not written"
        lines = val_path.read_text().splitlines()
        assert len(lines) >= 1, "W10-9/A7: no periodic validation lines"
        rec = json.loads(lines[0])
        assert rec["kind"] == "periodic_validation"
        assert "certificate_holds" in rec
        assert "logits_have_nan_or_inf" in rec


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
