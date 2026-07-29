"""aeon.job — W3 training-worker lifecycle: job dir, safe-stop protocol,
single-instance lock, launcher reattachment."""
from aeon.job.manager import (
    Job, JobStatus, create_job, load_job, active_jobs, safe_stop_request,
    request_emergency_terminate, is_stop_requested, mark_status,
)
from aeon.job.identity import (
    worker_identity, verify_worker_identity, WorkerIdentity,
)
from aeon.job.lock import SingleInstanceLock, LockAcquireError

__all__ = [
    "Job", "JobStatus", "create_job", "load_job", "active_jobs",
    "safe_stop_request", "request_emergency_terminate", "is_stop_requested",
    "mark_status", "worker_identity", "verify_worker_identity",
    "WorkerIdentity", "SingleInstanceLock", "LockAcquireError",
]
