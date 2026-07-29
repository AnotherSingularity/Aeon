"""
aeon/windows_paths.py — installed-resource + user-data path resolution.

Directive §5, §W4 defaults:
    Application binary:  %LOCALAPPDATA%\\Programs\\Aeon
    Runtime data root:   %LOCALAPPDATA%\\Aeon
    Config:              %LOCALAPPDATA%\\Aeon\\config
    Jobs:                %LOCALAPPDATA%\\Aeon\\jobs
    Logs:                %LOCALAPPDATA%\\Aeon\\logs
    Evidence:            %LOCALAPPDATA%\\Aeon\\evidence

On non-Windows (dev), we map:
    Runtime data root:   $XDG_DATA_HOME/Aeon  or  ~/.local/share/Aeon
    Application binary:  parent of this module (source-mode)

Frozen-vs-source detection:
    - sys.frozen True   → PyInstaller `_MEIPASS` gives installed resources
    - source            → resource root is the repository root

Every path is returned as an absolute POSIX-style string via os.fspath, and
directories are created on demand only in the WRITABLE roots.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional


APP_NAME = "Aeon"


# ---------------------------------------------------------------------------
# Frozen-vs-source detection
# ---------------------------------------------------------------------------
def is_frozen() -> bool:
    """True iff running from a PyInstaller (or similar) frozen bundle."""
    return bool(getattr(sys, "frozen", False))


def installed_resource_root() -> Path:
    """Read-only installed resources: configs/, docs/, manifests/, licenses/.

    Frozen: PyInstaller's `_MEIPASS` (extracted onedir root) OR the executable's
            containing directory for onedir mode.
    Source: the repository root (parent of the aeon/ package).
    """
    if is_frozen():
        # PyInstaller sets _MEIPASS on onefile; onedir uses the exe's dir.
        base = getattr(sys, "_MEIPASS", None)
        if base:
            return Path(base)
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def user_data_root() -> Path:
    """Writable per-user root. On Windows: %LOCALAPPDATA%\\Aeon.
    On POSIX: $XDG_DATA_HOME/Aeon or ~/.local/share/Aeon.
    Overridable via AEON_DATA_DIR (used by tests and dev configs)."""
    override = os.environ.get("AEON_DATA_DIR")
    if override:
        return Path(override).resolve()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.join(
            os.path.expanduser("~"), "AppData", "Local")
        return Path(base) / APP_NAME
    xdg = os.environ.get("XDG_DATA_HOME") or os.path.join(
        os.path.expanduser("~"), ".local", "share")
    return Path(xdg) / APP_NAME


def config_dir() -> Path:
    return user_data_root() / "config"


def jobs_dir() -> Path:
    return user_data_root() / "jobs"


def logs_dir() -> Path:
    return user_data_root() / "logs"


def evidence_dir() -> Path:
    return user_data_root() / "evidence"


def default_checkpoint_dir() -> Path:
    """Default checkpoint location if the user does not override."""
    return user_data_root() / "checkpoints"


def ensure_writable_layout() -> dict:
    """Create the writable directories on demand (never touches installed_resource_root).
    Returns a dict of the created paths for logging."""
    layout = {"user_data": user_data_root(), "config": config_dir(),
              "jobs": jobs_dir(), "logs": logs_dir(), "evidence": evidence_dir(),
              "checkpoints": default_checkpoint_dir()}
    for _, p in layout.items():
        p.mkdir(parents=True, exist_ok=True)
    return {k: str(v) for k, v in layout.items()}


def resolve_installed(relative: str) -> Path:
    """Resolve a repo-relative path (e.g. `configs/aeon_350m_primary.yaml`)
    against the installed_resource_root, never letting the CWD influence it."""
    if os.path.isabs(relative):
        return Path(relative)
    return installed_resource_root() / relative
