"""aeon/job/manager.py — job dir creation, safe-stop protocol, status tracking.

Every job under <jobs>/<job_id>/ contains:
    job.json          — configuration + identities (created on start)
    status.json       — latest structured status (updated by worker)
    metrics.jsonl     — always-on/sampled metrics (worker writes)
    events.jsonl      — hash-chained audit events (worker writes)
    worker.pid        — PID (raw; identity file has the trust binding)
    worker.identity   — worker fingerprint (JSON) written at worker start
    stop.request      — non-empty file: safe-stop requested by launcher
    stop.emergency    — non-empty file: emergency terminate requested
    result.json       — final job state (written by worker on clean exit)
"""
from __future__ import annotations

import enum
import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from aeon.windows_paths import jobs_dir


class JobStatus(str, enum.Enum):
    CREATED = "CREATED"
    STARTING = "STARTING"
    PREFLIGHT = "PREFLIGHT"
    RUNNING = "RUNNING"
    CHECKPOINTING = "CHECKPOINTING"
    STOP_REQUESTED = "STOP_REQUESTED"
    STOPPED = "STOPPED"
    FAILED = "FAILED"
    RECOVERY_REQUIRED = "RECOVERY_REQUIRED"


@dataclass
class Job:
    job_id: str
    job_dir: str
    config_path: str
    tokenizer_path: Optional[str]
    corpus_path: Optional[str]
    checkpoint_dir: str
    metrics_dir: str
    audit_dir: str
    runtime_policy_id: str
    security_policy_id: str
    checkpoint_policy: Dict[str, Any]
    created_at: float
    aeon_source_commit: str
    aeon_release: str
    # W10-3: distinct Start / Resume / Recovery intent. The worker consults
    # this instead of tcfg["resume"] (which was a config-file toggle that
    # could not distinguish the three GUI paths). "start" is the default and
    # means fresh training. "resume" means load the most recent authenticated
    # checkpoint from checkpoint_dir. "recover" means load a specific older
    # authenticated checkpoint under recovery_decision_path.
    intent: str = "start"
    resume_from_checkpoint: Optional[str] = None
    recovery_decision_path: Optional[str] = None
    # W10-9/A7: launcher-configured worker settings that the worker used to
    # ignore (cpu_thread_limit, memory_ceiling_gb, resume_preference). Kept
    # optional with a default of {} so pre-W10-9 job.json files still load.
    compute_policy: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def job_json_path(self) -> str:
        return os.path.join(self.job_dir, "job.json")

    @property
    def status_json_path(self) -> str:
        return os.path.join(self.job_dir, "status.json")

    @property
    def stop_request_path(self) -> str:
        return os.path.join(self.job_dir, "stop.request")

    @property
    def stop_emergency_path(self) -> str:
        return os.path.join(self.job_dir, "stop.emergency")

    @property
    def worker_pid_path(self) -> str:
        return os.path.join(self.job_dir, "worker.pid")

    @property
    def worker_identity_path(self) -> str:
        return os.path.join(self.job_dir, "worker.identity")

    @property
    def result_json_path(self) -> str:
        return os.path.join(self.job_dir, "result.json")


def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, sort_keys=True, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def create_job(
    *,
    config_path: str,
    tokenizer_path: Optional[str],
    corpus_path: Optional[str],
    checkpoint_dir: str,
    metrics_dir: str,
    audit_dir: str,
    checkpoint_policy: Dict[str, Any],
    runtime_policy_id: str = "aeon-runtime-v1",
    security_policy_id: str = "aeon-security-v1",
    intent: str = "start",
    resume_from_checkpoint: Optional[str] = None,
    recovery_decision_path: Optional[str] = None,
    compute_policy: Optional[Dict[str, Any]] = None,
) -> Job:
    if intent not in ("start", "resume", "recover"):
        raise ValueError(f"intent must be 'start'|'resume'|'recover', got {intent!r}")
    if intent == "recover" and not recovery_decision_path:
        raise ValueError("recover intent requires recovery_decision_path")
    from aeon.version import RELEASE_METADATA
    job_id = uuid.uuid4().hex
    job_dir = str(jobs_dir() / job_id)
    Path(job_dir).mkdir(parents=True, exist_ok=True)
    for d in (checkpoint_dir, metrics_dir, audit_dir):
        Path(d).mkdir(parents=True, exist_ok=True)
    job = Job(
        job_id=job_id, job_dir=job_dir,
        config_path=os.fspath(config_path),
        tokenizer_path=os.fspath(tokenizer_path) if tokenizer_path else None,
        corpus_path=os.fspath(corpus_path) if corpus_path else None,
        checkpoint_dir=os.fspath(checkpoint_dir),
        metrics_dir=os.fspath(metrics_dir),
        audit_dir=os.fspath(audit_dir),
        runtime_policy_id=runtime_policy_id,
        security_policy_id=security_policy_id,
        checkpoint_policy=dict(checkpoint_policy),
        created_at=time.time(),
        aeon_source_commit=RELEASE_METADATA.get("source_commit", "unknown"),
        aeon_release=RELEASE_METADATA.get("semantic_version", "unknown"),
        intent=intent,
        resume_from_checkpoint=(os.fspath(resume_from_checkpoint)
                                 if resume_from_checkpoint else None),
        recovery_decision_path=(os.fspath(recovery_decision_path)
                                 if recovery_decision_path else None),
        compute_policy=dict(compute_policy or {}),
    )
    _atomic_write_json(job.job_json_path, job.to_dict())
    mark_status(job, JobStatus.CREATED, note=f"job created (intent={intent})")
    return job


def load_job(job_dir_or_json: str) -> Optional[Job]:
    p = Path(job_dir_or_json)
    if p.is_dir():
        p = p / "job.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    # W10-3 backward compatibility: pre-W10-3 job.json files have no
    # intent / resume_from_checkpoint / recovery_decision_path fields.
    # Treat those as intent="start" — the safest disposition.
    data.setdefault("intent", "start")
    data.setdefault("resume_from_checkpoint", None)
    data.setdefault("recovery_decision_path", None)
    # W10-9 backward compatibility: pre-W10-9 job.json has no compute_policy.
    data.setdefault("compute_policy", {})
    return Job(**data)


def active_jobs() -> List[Job]:
    root = jobs_dir()
    if not root.exists():
        return []
    out: List[Job] = []
    for sub in sorted(root.iterdir()):
        if sub.is_dir():
            j = load_job(str(sub))
            if j is not None:
                out.append(j)
    return out


def mark_status(job: Job, status: JobStatus, *, note: str = "", **extras) -> None:
    rec = {"status": str(status.value), "ts": time.time(), "note": note}
    rec.update(extras)
    _atomic_write_json(job.status_json_path, rec)


def safe_stop_request(job: Job, *, reason: str = "user requested safe stop") -> None:
    """Create the stop.request file with a nonce + reason. The worker polls
    this file at safe checkpoint boundaries."""
    nonce = uuid.uuid4().hex
    payload = {"nonce": nonce, "reason": reason, "ts": time.time()}
    _atomic_write_json(job.stop_request_path, payload)


def request_emergency_terminate(job: Job) -> None:
    """Create the stop.emergency marker. Launcher UI must gate this behind an
    explicit user confirmation."""
    _atomic_write_json(job.stop_emergency_path,
                        {"ts": time.time(), "reason": "emergency terminate"})


def is_stop_requested(job: Job) -> bool:
    return os.path.exists(job.stop_request_path)


def is_emergency_terminate_requested(job: Job) -> bool:
    return os.path.exists(job.stop_emergency_path)
