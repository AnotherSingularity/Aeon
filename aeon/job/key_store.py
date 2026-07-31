"""aeon/job/key_store.py — per-job HMAC key material for W10-2.

Provides the ``KeyRef`` the GUI worker passes to
``aeon.protected_checkpoint.protected_save``/``protected_load`` so that
Safe Stop actually produces the authenticated envelope the launcher and W9
documentation claim.

Trust posture (documented, not hidden):

  A key stored under ``<job_dir>/hmac.key`` is a **development integrity**
  trust root. It authenticates the checkpoint against accidental corruption
  and against a second worker on the same job dir stamping over the
  envelope. It does NOT authenticate the checkpoint against an adversary
  who can read the job directory: such an adversary can read the key file
  and forge a matching MAC. That threat model is closed by W10-6 (signed
  manifest / trusted root) and by the operator's OS-level access control.

  This posture is honest and matches the current unsigned Windows build:
  an installer without Authenticode cannot make claims stronger than
  filesystem-level protection anyway. When W10-6 introduces a signed
  release-time trust root, the key can be derived from that root instead
  of being written to disk. The W10-2 API here does not change.

Never included in the runtime manifest, never printed to logs, never
committed to the repo. The file is opened with restrictive permissions
where the operating system supports them (POSIX 0600). On Windows the
per-user ``%LOCALAPPDATA%\\Aeon\\jobs\\<id>\\`` directory is already
per-user by construction; W10-9 will migrate to DPAPI if the operator's
policy requires stronger sealing.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path

from aeon.protected_checkpoint import KeyRef


HMAC_KEY_LEN = 32
KEY_FILENAME = "hmac.key"


class KeyStoreError(RuntimeError):
    pass


def _restrict_file_permissions(path: str) -> None:
    """Best-effort tighten to owner-only where the OS supports it. Silently
    accept a no-op on Windows — the parent dir is already per-user."""
    try:
        os.chmod(path, 0o600)
    except Exception:
        pass


def _key_path(job_dir: str) -> str:
    return os.path.join(job_dir, KEY_FILENAME)


def ensure_job_hmac_keyref(job_dir: str, *, allow_create: bool = True) -> KeyRef:
    """Return a KeyRef for this job's HMAC key. Generates the key on first
    call and reuses it thereafter so that resume across restarts works.

    Raises ``KeyStoreError`` if the key does not exist and ``allow_create``
    is False, or if the key file cannot be read or has an unexpected size.

    Parameters
    ----------
    job_dir:
        The job's ``job_dir`` — the same directory containing job.json,
        status.json, worker.identity, etc.
    allow_create:
        Defaults True at worker startup so the first checkpoint has a key.
        Callers verifying an existing chain (protected_load) may set False
        if they want the operation to fail closed rather than silently
        forge a new key.
    """
    kp = _key_path(job_dir)
    if os.path.exists(kp):
        try:
            data = Path(kp).read_bytes()
        except Exception as e:
            raise KeyStoreError(f"cannot read {kp!r}: {e}") from e
        if len(data) != HMAC_KEY_LEN:
            raise KeyStoreError(
                f"{kp!r} has wrong length ({len(data)} bytes, expected {HMAC_KEY_LEN})")
        key = data
    else:
        if not allow_create:
            raise KeyStoreError(f"HMAC key absent at {kp!r} and allow_create=False")
        Path(job_dir).mkdir(parents=True, exist_ok=True)
        key = secrets.token_bytes(HMAC_KEY_LEN)
        # Write atomically to avoid a torn key file on crash mid-write.
        tmp = kp + ".tmp"
        with open(tmp, "wb") as fh:
            fh.write(key)
        os.replace(tmp, kp)
        _restrict_file_permissions(kp)
    # KeyRef.resolve() must return the raw bytes; store `key` in the closure
    # so it survives after this function returns.
    handle = f"job:{os.path.basename(os.path.abspath(job_dir))}"
    return KeyRef(handle=handle, resolve=lambda: key)
