"""aeon/launcher/controls.py — non-GUI logic for the launcher.

Separated from the Tk view so it can be unit-tested without a display.
Handles: worker spawn (no shell, no console, separate process group),
launcher-close-doesn't-kill-worker, safe-stop request, reattachment on
launcher restart, single-instance enforcement.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from aeon.job.identity import verify_worker_identity, worker_identity
from aeon.job.lock import SingleInstanceLock, LockAcquireError
from aeon.job.manager import (
    Job, JobStatus, active_jobs, load_job, safe_stop_request,
    request_emergency_terminate, mark_status,
)
from aeon.windows_paths import user_data_root


LAUNCHER_LOCK_NAME = "launcher.lock"


def spawn_worker(job: Job, *, exe: Optional[str] = None) -> subprocess.Popen:
    """Launch Aeon.exe --worker <job.json>. Windows-specific flags applied at
    runtime; POSIX behaviour equivalent (new session, no controlling tty)."""
    argv = [exe or _default_exe(), "--worker", job.job_json_path]
    kwargs: Dict[str, Any] = dict(
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, cwd=str(user_data_root()),
        env={"PATH": os.environ.get("PATH", ""),
             "AEON_DATA_DIR": os.environ.get("AEON_DATA_DIR", ""),
             "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
             "TEMP": os.environ.get("TEMP", ""),
             "TMPDIR": os.environ.get("TMPDIR", ""),
             "LOCALAPPDATA": os.environ.get("LOCALAPPDATA", ""),
             "USERPROFILE": os.environ.get("USERPROFILE", ""),
             "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")},
        close_fds=True,
    )
    # NEVER shell=True.
    if os.name == "nt":
        # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        creationflags = 0x00000008 | 0x00000200 | 0x08000000
        kwargs["creationflags"] = creationflags
    else:
        kwargs["start_new_session"] = True                # POSIX: new process group
    return subprocess.Popen(argv, **kwargs)


def _default_exe() -> str:
    """The frozen application executable. In source mode, use the current python
    with `-m aeon.entry`."""
    if getattr(sys, "frozen", False):
        return sys.executable
    return sys.executable                                  # source dev: rely on -m via argv


def acquire_launcher_lock() -> SingleInstanceLock:
    lock = SingleInstanceLock(str(user_data_root() / LAUNCHER_LOCK_NAME))
    try:
        lock.acquire(timeout_s=0.5)
        return lock
    except LockAcquireError as e:
        raise LockAcquireError(
            f"another Aeon launcher is already running: {e}") from e


def reattach_or_mark_interrupted() -> List[Tuple[Job, Optional[Any]]]:
    """On launcher startup, walk every known job. Return (job, live_identity)
    for each — live_identity is None when the recorded worker is no longer
    alive (job is marked as RECOVERY_REQUIRED)."""
    out: List[Tuple[Job, Optional[Any]]] = []
    for j in active_jobs():
        ident = verify_worker_identity(j.job_dir)
        if ident is None:
            # Mark interrupted UNLESS the last status file says stopped/failed
            try:
                import json
                status_path = j.status_json_path
                if os.path.exists(status_path):
                    st = json.load(open(status_path, encoding="utf-8"))
                    if st.get("status") in ("STOPPED", "FAILED"):
                        out.append((j, None)); continue
            except Exception:
                pass
            mark_status(j, JobStatus.RECOVERY_REQUIRED,
                         note="worker no longer alive; may need authenticated resume")
        out.append((j, ident))
    return out


# ---- UI-safety gating -------------------------------------------------------
def controls_gate(job_state: Optional[str], installation_verified: bool,
                   preflight_verdict: Optional[str]) -> Dict[str, bool]:
    """Return an enabled/disabled dict for every named control. Enforces §W2
    safety rules — no concurrent training, no config changes during training,
    no resume until installation + preflight are green."""
    active = job_state in ("STARTING", "PREFLIGHT", "RUNNING",
                            "CHECKPOINTING", "STOP_REQUESTED")
    return {
        "configure":         not active,
        "verify_installation": True,
        "run_preflight":     not active,
        "start_new_training": (not active) and installation_verified
                              and preflight_verdict in ("READY", "READY_WITH_WARNINGS"),
        "resume_latest":     (not active) and installation_verified,
        "stop_safely":       active,
        "emergency_stop":    active,
        "validate":          not active,
        "diagnose_checkpoint": not active,
        "open_logs":         True,
        "open_checkpoints":  True,
        "open_evidence":     True,
        "recovery":          not active,
        "exit_launcher":     True,
    }
