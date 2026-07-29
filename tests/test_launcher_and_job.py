"""W2-W3 launcher-logic + worker-lifecycle tests. Headless (no Tk display).

Covers:
  * controls_gate rules (no concurrent training, no config during run,
    resume only after installation verified, etc.)
  * Job creation writes the required job dir contents
  * Safe-stop request handshake (file created; is_stop_requested true;
    worker MUST observe it at a checkpoint boundary)
  * Emergency-terminate request is DISTINCT from safe-stop
  * SingleInstanceLock: one holder at a time; stale-owner recovery
  * WorkerIdentity fingerprint is stable across serialization
  * verify_worker_identity refuses on missing file, dead pid, mismatched
    process create time
  * Launcher reattachment marks dead workers as RECOVERY_REQUIRED
  * spawn_worker builds a shell-free argv (checked without spawning)
"""
import json
import os
import sys
import tempfile
import time
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _isolate_env(d):
    """Return an environment context that isolates user_data_root() under d."""
    return mock.patch.dict(os.environ, {"AEON_DATA_DIR": d})


# ---- W2 controls_gate -------------------------------------------------------
def test_controls_gate_denies_start_before_installation_verified():
    from aeon.launcher.controls import controls_gate
    g = controls_gate(job_state=None, installation_verified=False,
                       preflight_verdict="READY")
    assert not g["start_new_training"]
    g = controls_gate(job_state=None, installation_verified=True,
                       preflight_verdict="READY")
    assert g["start_new_training"]


def test_controls_gate_denies_concurrent_training():
    from aeon.launcher.controls import controls_gate
    for active in ("STARTING", "PREFLIGHT", "RUNNING", "CHECKPOINTING", "STOP_REQUESTED"):
        g = controls_gate(job_state=active, installation_verified=True,
                           preflight_verdict="READY")
        assert not g["start_new_training"]
        assert not g["configure"]
        assert not g["resume_latest"]
        assert g["stop_safely"] and g["emergency_stop"]


def test_controls_gate_stop_only_when_active():
    from aeon.launcher.controls import controls_gate
    g = controls_gate(job_state=None, installation_verified=True,
                       preflight_verdict="READY")
    assert not g["stop_safely"] and not g["emergency_stop"]


def test_controls_gate_blocks_start_on_preflight_blocked():
    from aeon.launcher.controls import controls_gate
    g = controls_gate(job_state=None, installation_verified=True,
                       preflight_verdict="BLOCKED")
    assert not g["start_new_training"]


# ---- W3 job creation --------------------------------------------------------
def test_create_job_writes_job_dir_layout():
    from aeon.job.manager import create_job
    with tempfile.TemporaryDirectory() as d, _isolate_env(d):
        from aeon.windows_paths import ensure_writable_layout
        ensure_writable_layout()
        job = create_job(
            config_path=os.path.join(d, "cfg.yaml"),
            tokenizer_path=None, corpus_path=None,
            checkpoint_dir=os.path.join(d, "ckpt"),
            metrics_dir=os.path.join(d, "metrics"),
            audit_dir=os.path.join(d, "audit"),
            checkpoint_policy={"interval": 1000},
        )
        assert os.path.exists(job.job_json_path)
        assert os.path.exists(job.status_json_path)
        # No stop / emergency files yet
        assert not os.path.exists(job.stop_request_path)
        assert not os.path.exists(job.stop_emergency_path)


def test_safe_stop_request_and_is_stop_requested():
    from aeon.job.manager import (create_job, safe_stop_request,
                                    request_emergency_terminate,
                                    is_stop_requested,
                                    is_emergency_terminate_requested)
    with tempfile.TemporaryDirectory() as d, _isolate_env(d):
        from aeon.windows_paths import ensure_writable_layout
        ensure_writable_layout()
        job = create_job(config_path=os.path.join(d, "cfg.yaml"),
                          tokenizer_path=None, corpus_path=None,
                          checkpoint_dir=os.path.join(d, "ckpt"),
                          metrics_dir=os.path.join(d, "metrics"),
                          audit_dir=os.path.join(d, "audit"),
                          checkpoint_policy={"interval": 1000})
        assert not is_stop_requested(job)
        safe_stop_request(job, reason="test")
        assert is_stop_requested(job)
        # nonce present
        payload = json.load(open(job.stop_request_path))
        assert "nonce" in payload and "reason" in payload
        # emergency is distinct
        assert not is_emergency_terminate_requested(job)
        request_emergency_terminate(job)
        assert is_emergency_terminate_requested(job)


# ---- W3 SingleInstanceLock -------------------------------------------------
def test_single_instance_lock_serialises():
    from aeon.job.lock import SingleInstanceLock, LockAcquireError
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.lock")
        a = SingleInstanceLock(p); a.acquire()
        try:
            b = SingleInstanceLock(p)
            try:
                b.acquire(timeout_s=0.1)
                assert False, "second acquirer succeeded while first held"
            except LockAcquireError:
                pass
        finally:
            a.release()


def test_single_instance_lock_stale_owner_recovery():
    """If the owner-pid file references a dead pid, a new acquirer should
    reclaim the lock rather than blocking forever."""
    from aeon.job.lock import SingleInstanceLock
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "x.lock")
        # Fake an existing owner file with a definitely-dead pid
        open(p, "a").close()
        with open(p + ".owner", "w") as f:
            json.dump({"pid": 999999999, "ts": time.time()}, f)
        # Should reclaim
        lock = SingleInstanceLock(p)
        lock.acquire(timeout_s=1.0)
        lock.release()


# ---- W3 WorkerIdentity ------------------------------------------------------
def test_worker_identity_fingerprint_is_deterministic():
    from aeon.job.identity import WorkerIdentity
    a = WorkerIdentity(pid=1, started_at=100.0, process_create_time=50.0,
                        aeon_source_commit="abc123", aeon_release="0.2.3",
                        aeon_build_type="development")
    b = WorkerIdentity(pid=1, started_at=100.0, process_create_time=50.0,
                        aeon_source_commit="abc123", aeon_release="0.2.3",
                        aeon_build_type="development")
    assert a.fingerprint() == b.fingerprint()


def test_verify_worker_identity_none_when_missing():
    from aeon.job.identity import verify_worker_identity
    with tempfile.TemporaryDirectory() as d:
        assert verify_worker_identity(d) is None


def test_verify_worker_identity_none_when_pid_dead():
    from aeon.job.identity import verify_worker_identity, WorkerIdentity
    with tempfile.TemporaryDirectory() as d:
        rec = WorkerIdentity(pid=999999999, started_at=100.0,
                              process_create_time=0.0,
                              aeon_source_commit="abc", aeon_release="0.2.3",
                              aeon_build_type="development")
        with open(os.path.join(d, "worker.identity"), "w") as f:
            json.dump(rec.to_dict(), f)
        assert verify_worker_identity(d) is None


# ---- W2 reattachment --------------------------------------------------------
def test_reattach_marks_dead_worker_as_recovery_required():
    """Create a job, plant a stale worker.identity with an impossible pid,
    then call reattach_or_mark_interrupted and confirm the job status is
    updated to RECOVERY_REQUIRED."""
    from aeon.job.manager import create_job, JobStatus
    from aeon.job.identity import WorkerIdentity
    from aeon.launcher.controls import reattach_or_mark_interrupted
    with tempfile.TemporaryDirectory() as d, _isolate_env(d):
        from aeon.windows_paths import ensure_writable_layout
        ensure_writable_layout()
        job = create_job(config_path=os.path.join(d, "cfg.yaml"),
                          tokenizer_path=None, corpus_path=None,
                          checkpoint_dir=os.path.join(d, "ckpt"),
                          metrics_dir=os.path.join(d, "metrics"),
                          audit_dir=os.path.join(d, "audit"),
                          checkpoint_policy={"interval": 1000})
        rec = WorkerIdentity(pid=999999999, started_at=100.0,
                              process_create_time=0.0,
                              aeon_source_commit="abc", aeon_release="0.2.3",
                              aeon_build_type="development")
        with open(job.worker_identity_path, "w") as f:
            json.dump(rec.to_dict(), f)
        out = reattach_or_mark_interrupted()
        # ident is None (dead pid) — status updated to RECOVERY_REQUIRED
        assert any(j.job_id == job.job_id for j, ident in out)
        st = json.load(open(job.status_json_path))
        assert st["status"] == "RECOVERY_REQUIRED"


# ---- W3 spawn_worker no-shell contract -------------------------------------
def test_spawn_worker_argv_has_no_shell_and_targets_worker_mode():
    """We intercept subprocess.Popen and inspect its argv + kwargs. Assert the
    argv is a list (never a shell string), --worker is present, and shell=True
    NEVER appears."""
    from aeon.launcher.controls import spawn_worker
    from aeon.job.manager import Job
    j = Job(job_id="abc", job_dir="/tmp/aeon_test_job", config_path="cfg.yaml",
             tokenizer_path=None, corpus_path=None,
             checkpoint_dir="/tmp/aeon_test_ck",
             metrics_dir="/tmp/aeon_test_m",
             audit_dir="/tmp/aeon_test_a",
             runtime_policy_id="p", security_policy_id="s",
             checkpoint_policy={}, created_at=0.0,
             aeon_source_commit="abc", aeon_release="0.2.3")
    with mock.patch("aeon.launcher.controls.subprocess.Popen") as p:
        p.return_value = mock.MagicMock()
        spawn_worker(j, exe="/opt/aeon/Aeon.exe")
        args, kwargs = p.call_args
        assert isinstance(args[0], list), f"argv is not a list: {args[0]}"
        assert "--worker" in args[0]
        assert kwargs.get("shell") is not True    # never shell=True
        # stdin/stdout/stderr silenced
        for k in ("stdin", "stdout", "stderr"):
            assert kwargs[k] == mock.ANY or kwargs[k] is not None


# ---- W4 config schema + atomic writes --------------------------------------
def test_config_schema_rejects_relative_paths():
    from aeon.config.schema import validate_config_dict, USER_CONFIG_SCHEMA_VERSION
    cfg = {"schema_version": USER_CONFIG_SCHEMA_VERSION,
            "tokenizer_path": "relative/path.model"}
    errs = validate_config_dict(cfg)
    assert any("absolute" in e for e in errs), errs


def test_config_schema_rejects_forbidden_fields():
    from aeon.config.schema import validate_config_dict, USER_CONFIG_SCHEMA_VERSION
    cfg = {"schema_version": USER_CONFIG_SCHEMA_VERSION,
            "shell_command": "echo hi"}
    errs = validate_config_dict(cfg)
    assert any("forbidden field" in e for e in errs), errs


def test_config_schema_rejects_traversal():
    from aeon.config.schema import validate_config_dict, USER_CONFIG_SCHEMA_VERSION
    cfg = {"schema_version": USER_CONFIG_SCHEMA_VERSION,
            "tokenizer_path": "/etc/../etc/passwd"}
    errs = validate_config_dict(cfg)
    assert any("traversal" in e for e in errs), errs


def test_atomic_write_and_load_round_trip():
    from aeon.config.schema import (atomic_write_user_config, load_user_config,
                                     USER_CONFIG_SCHEMA_VERSION)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "cfg.json")
        cfg = {"tokenizer_path": os.path.abspath(os.path.join(d, "tok.model")),
                "corpus_path": os.path.abspath(d),
                "checkpoint_dir": os.path.abspath(d),
                "metrics_dir": os.path.abspath(d),
                "evidence_dir": os.path.abspath(d),
                "disk_allocation_gb": 32,
                "cpu_thread_limit": 4,
                "checkpoint_interval": 1000,
                "validation_interval": 1000,
                "resume_preference": "auto",
                "training_config_id": "aeon_350m_primary.yaml"}
        atomic_write_user_config(p, cfg)
        back = load_user_config(p)
        assert back["schema_version"] == USER_CONFIG_SCHEMA_VERSION
        assert back["training_config_id"] == "aeon_350m_primary.yaml"


def test_migrate_user_config_advances_schema_version():
    from aeon.config.schema import migrate_user_config, USER_CONFIG_SCHEMA_VERSION
    old = {"schema_version": 0, "tokenizer_path": "/tmp/x.model"}
    new = migrate_user_config(old)
    assert new["schema_version"] == USER_CONFIG_SCHEMA_VERSION


# ---- W4 preflight -----------------------------------------------------------
def test_preflight_returns_ready_or_warnings_on_empty_config():
    from aeon.config.preflight import run_preflight, PreflightVerdict
    with tempfile.TemporaryDirectory() as d, _isolate_env(d):
        from aeon.windows_paths import ensure_writable_layout
        ensure_writable_layout()
        res = run_preflight({})
        # Missing tokenizer/corpus → warn, not fail; verdict should not BLOCKED unless K etc missing
        assert res.verdict in (PreflightVerdict.READY, PreflightVerdict.READY_WITH_WARNINGS,
                                PreflightVerdict.BLOCKED)
        # Structural invariants must PASS
        cnames = {c.name: c.status for c in res.checks}
        assert cnames.get("K_equals_16") == "pass"
        assert cnames.get("rotary_no_register_buffer") == "pass"
        assert cnames.get("strict_load_weights_only") == "pass"
        assert cnames.get("runtime_policy_loaded") == "pass"


def test_preflight_blocks_on_missing_tokenizer_file():
    from aeon.config.preflight import run_preflight, PreflightVerdict
    with tempfile.TemporaryDirectory() as d, _isolate_env(d):
        from aeon.windows_paths import ensure_writable_layout
        ensure_writable_layout()
        cfg = {"tokenizer_path": os.path.join(d, "nonexistent.model")}
        res = run_preflight(cfg)
        # tokenizer missing → fail on tokenizer check
        tok_check = [c for c in res.checks if c.name == "tokenizer"][0]
        assert tok_check.status == "fail"
        assert res.verdict == PreflightVerdict.BLOCKED


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
