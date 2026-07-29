"""aeon/job/identity.py — worker-identity fingerprint (§W3 stale-process handling).

Windows may reuse PIDs. Trusting `os.kill(pid, 0) == 0` alone is unsafe. We
combine PID + process-start-time + Aeon release identity into an opaque
worker.identity file. On reattachment the launcher recomputes the identity for
the recorded PID and refuses reattachment on mismatch."""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from aeon.version import RELEASE_METADATA


@dataclass
class WorkerIdentity:
    pid: int
    started_at: float                    # wall clock at worker start
    process_create_time: Optional[float] # OS-reported start time when available
    aeon_source_commit: str
    aeon_release: str
    aeon_build_type: str

    def to_dict(self) -> dict:
        return asdict(self)

    def fingerprint(self) -> str:
        h = hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
                                       separators=(",", ":")).encode()).hexdigest()
        return h[:32]


def _process_create_time(pid: int) -> Optional[float]:
    """Return the OS-reported process create-time in seconds since epoch, or None.
    Uses /proc/<pid>/stat on Linux, GetProcessTimes on Windows via ctypes."""
    if not pid or pid <= 0:
        return None
    if os.name == "posix":
        try:
            stat_path = f"/proc/{pid}/stat"
            with open(stat_path) as fh:
                data = fh.read()
            # field 22 (1-indexed) is starttime in clock ticks since boot
            # Get the substring after the executable name in parens
            r = data.rfind(")")
            if r < 0:
                return None
            fields = data[r + 2:].split()
            ticks = int(fields[19])                     # field 22 is index 19 after skipping first two
            hz = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
            try:
                boot_ns = None
                with open("/proc/stat") as fh:
                    for line in fh:
                        if line.startswith("btime "):
                            boot_ns = int(line.split()[1])
                            break
                if boot_ns is not None:
                    return boot_ns + (ticks / hz)
            except Exception:
                pass
            return ticks / hz
        except Exception:
            return None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            OpenProcess = k32.OpenProcess
            OpenProcess.restype = wintypes.HANDLE
            OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            CloseHandle = k32.CloseHandle
            CloseHandle.argtypes = [wintypes.HANDLE]
            GetProcessTimes = k32.GetProcessTimes
            GetProcessTimes.argtypes = [wintypes.HANDLE,
                                         ctypes.POINTER(wintypes.FILETIME),
                                         ctypes.POINTER(wintypes.FILETIME),
                                         ctypes.POINTER(wintypes.FILETIME),
                                         ctypes.POINTER(wintypes.FILETIME)]
            GetProcessTimes.restype = wintypes.BOOL
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if not h:
                return None
            try:
                creation = wintypes.FILETIME()
                exit_ = wintypes.FILETIME()
                kernel = wintypes.FILETIME()
                user = wintypes.FILETIME()
                if not GetProcessTimes(h, ctypes.byref(creation), ctypes.byref(exit_),
                                        ctypes.byref(kernel), ctypes.byref(user)):
                    return None
                # FILETIME is 100-nanosecond intervals since 1601-01-01.
                filetime = (creation.dwHighDateTime << 32) | creation.dwLowDateTime
                # Convert to seconds since epoch (1970-01-01)
                return (filetime - 116444736000000000) / 10_000_000
            finally:
                CloseHandle(h)
        except Exception:
            return None
    return None


def worker_identity(pid: Optional[int] = None) -> WorkerIdentity:
    pid = pid if pid is not None else os.getpid()
    return WorkerIdentity(
        pid=pid,
        started_at=time.time(),
        process_create_time=_process_create_time(pid),
        aeon_source_commit=RELEASE_METADATA.get("source_commit", "unknown"),
        aeon_release=RELEASE_METADATA.get("semantic_version", "unknown"),
        aeon_build_type=RELEASE_METADATA.get("build_type", "development"),
    )


def _pid_alive(pid: int) -> bool:
    """OS-agnostic liveness check. On POSIX: os.kill(pid, 0). On Windows:
    OpenProcess."""
    if not pid or pid <= 0:
        return False
    if os.name == "posix":
        try:
            os.kill(pid, 0); return True
        except OSError:
            return False
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes
            k32 = ctypes.WinDLL("kernel32", use_last_error=True)
            OpenProcess = k32.OpenProcess
            OpenProcess.restype = wintypes.HANDLE
            OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            CloseHandle = k32.CloseHandle
            CloseHandle.argtypes = [wintypes.HANDLE]
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if h:
                CloseHandle(h); return True
        except Exception:
            pass
    return False


def verify_worker_identity(job_dir: str) -> Optional[WorkerIdentity]:
    """Load the recorded worker.identity from the job dir and re-verify it
    against the current process at that PID. Returns the recorded identity if
    the process still corresponds to the same Aeon release AND the same start
    time (guarding against PID reuse); returns None otherwise."""
    ident_path = Path(job_dir) / "worker.identity"
    if not ident_path.exists():
        return None
    try:
        data = json.loads(ident_path.read_text(encoding="utf-8"))
        recorded = WorkerIdentity(**data)
    except Exception:
        return None
    if not _pid_alive(recorded.pid):
        return None
    live_start = _process_create_time(recorded.pid)
    # Guard against PID reuse: recorded process_create_time must match within
    # a small tolerance (some OS's snap to a low-resolution clock).
    if recorded.process_create_time is not None and live_start is not None:
        if abs(live_start - recorded.process_create_time) > 2.0:
            return None
    # Aeon-release match
    if recorded.aeon_source_commit != RELEASE_METADATA.get("source_commit", "unknown"):
        # A different Aeon build may have started at the same PID; refuse.
        return None
    return recorded
