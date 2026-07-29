"""aeon/integrity.py — installed runtime-manifest verification (W5/W7).

The Windows build (W5) produces `packaging/windows/RUNTIME_MANIFEST.json`
containing per-file relative path + size + sha256 for every immutable-bundle
file. This module verifies the manifest against the on-disk installation.

Fail-closed: any missing file, any hash mismatch, any missing manifest → False
plus a structured report.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Tuple

from aeon.windows_paths import installed_resource_root


MANIFEST_RELATIVE = "packaging/windows/RUNTIME_MANIFEST.json"


def _sha256_file(path: str, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_manifest():
    path = installed_resource_root() / MANIFEST_RELATIVE
    if not path.exists():
        return None, {"error": "manifest_missing", "path": str(path)}
    try:
        return json.load(open(path, encoding="utf-8")), None
    except Exception as e:
        return None, {"error": "manifest_unreadable", "detail": str(e)}


def verify_installed_manifest() -> Tuple[bool, dict]:
    """Return (ok, report). Report includes per-file check counts + any failures."""
    manifest, err = _load_manifest()
    if manifest is None:
        return False, {"ok": False, "reason": err}
    root = installed_resource_root()
    files = manifest.get("files", [])
    missing: list = []
    mismatched: list = []
    ok_count = 0
    for entry in files:
        rel = entry.get("path")
        expected = entry.get("sha256")
        if not rel or not expected:
            continue
        full = root / rel
        if not full.exists():
            missing.append(rel)
            continue
        actual = _sha256_file(str(full))
        if actual != expected:
            mismatched.append({"path": rel, "expected": expected, "actual": actual})
        else:
            ok_count += 1
    ok = (not missing) and (not mismatched)
    return ok, {
        "ok": bool(ok), "files_checked": len(files),
        "files_ok": ok_count, "missing": missing, "mismatched": mismatched,
        "manifest_path": str(root / MANIFEST_RELATIVE),
        "installation_root": str(root),
    }
