"""aeon/job/lock.py — cross-platform single-instance lock.

Windows: msvcrt.locking on a lock file. POSIX: fcntl.flock. Both handle
stale locks by writing an owner-pid file alongside — a new acquirer whose
owner is dead reclaims the lock. The lock is meant to be held for the
lifetime of the launcher / worker process."""
from __future__ import annotations

import errno
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional


class LockAcquireError(RuntimeError):
    pass


class SingleInstanceLock:
    """Held for the caller's process lifetime. Exclusive by name."""

    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def _write_owner(self):
        try:
            with open(str(self.path) + ".owner", "w", encoding="utf-8") as f:
                json.dump({"pid": os.getpid(), "ts": time.time()}, f)
        except Exception:
            pass

    def _stale_owner(self) -> bool:
        owner_path = str(self.path) + ".owner"
        if not os.path.exists(owner_path):
            return True
        try:
            data = json.load(open(owner_path, encoding="utf-8"))
            pid = int(data.get("pid", 0))
        except Exception:
            return True
        # Reuse the identity-module liveness check
        try:
            from aeon.job.identity import _pid_alive
            return not _pid_alive(pid)
        except Exception:
            return True

    def acquire(self, timeout_s: float = 0.0) -> None:
        deadline = time.time() + max(0.0, timeout_s)
        while True:
            self._fh = open(self.path, "a+", encoding="utf-8")
            got = self._try_lock()
            if got:
                self._write_owner()
                return
            self._fh.close(); self._fh = None
            if self._stale_owner():
                try: os.unlink(self.path)
                except Exception: pass
                try: os.unlink(str(self.path) + ".owner")
                except Exception: pass
                continue
            if time.time() >= deadline:
                raise LockAcquireError(f"lock held by another process: {self.path}")
            time.sleep(0.1)

    def _try_lock(self) -> bool:
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(self._fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except (IOError, OSError) as e:
            if getattr(e, "errno", None) in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                return False
            return False

    def release(self) -> None:
        if self._fh is None:
            return
        try:
            if sys.platform == "win32":
                import msvcrt
                try:
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    pass
            else:
                import fcntl
                try: fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
                except Exception: pass
        finally:
            try: self._fh.close()
            except Exception: pass
            self._fh = None
            try: os.unlink(str(self.path) + ".owner")
            except Exception: pass

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False
