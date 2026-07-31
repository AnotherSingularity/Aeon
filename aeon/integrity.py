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
from typing import List, Tuple

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
    """Return (ok, report). Report includes per-file check counts + any
    failure lists.

    W10-6 corrections:

    * Malformed manifest entries (missing ``path`` or ``sha256``) FAIL the
      verification instead of being silently ``continue``d over (audit
      finding A10). A malformed entry is a defect in either the manifest
      generator or the installed manifest file itself; both are grounds to
      refuse an install.
    * Path traversal or absolute paths in manifest entries FAIL (defensive
      — the generator never emits them, but the verifier no longer trusts
      the manifest to be sane by construction).
    * Unexpected extra files inside the installed tree — files present on
      disk that the manifest does NOT list — FAIL (audit finding A11).
      The audit's specific concern was that an attacker could drop an
      extra .exe or .dll into the installed tree without detection.
    * Top-level files (``../Aeon.exe``, ``../python311.dll``) are covered
      when the manifest carries ``scope: top_level`` entries (audit
      finding A9). The manifest schema version is now recorded and enforced.
    """
    manifest, err = _load_manifest()
    if manifest is None:
        return False, {"ok": False, "reason": err}
    root = installed_resource_root()
    files = manifest.get("files", [])

    malformed: list = []
    missing: list = []
    mismatched: list = []
    ok_count = 0
    listed_paths: set = set()

    # Manifest schema version — schema 2+ carries scope, schema 1 doesn't.
    _schema = int(manifest.get("manifest_schema_version") or 1)

    for entry in files:
        rel = entry.get("path")
        expected = entry.get("sha256")
        if not rel or not expected:
            malformed.append({"entry": entry, "reason": "missing path or sha256"})
            continue
        if os.path.isabs(rel) or ".." in rel.split("/") and not rel.startswith("../"):
            # A bare ".." mid-path is traversal. A leading "../" is the
            # documented scope="top_level" convention introduced by W10-6.
            malformed.append({"entry": entry, "reason": "unsafe relative path"})
            continue
        full = (root / rel).resolve()
        # Ensure the resolved path stays within root OR is a documented
        # top-level file (one level up from root).
        try:
            full.relative_to(root.resolve())
        except ValueError:
            # Might be a legitimate top-level "../<name>" entry — verify
            # that the path stays inside root.parent (the bundle root).
            try:
                full.relative_to(root.resolve().parent)
            except ValueError:
                malformed.append({"entry": entry, "reason": "resolves outside bundle"})
                continue
        listed_paths.add(str(full))
        if not full.exists():
            missing.append(rel)
            continue
        actual = _sha256_file(str(full))
        if actual != expected:
            mismatched.append({"path": rel, "expected": expected, "actual": actual})
        else:
            ok_count += 1

    # Walk the installed tree looking for files the manifest does NOT list.
    # Only executable / DLL / policy / configuration surfaces are actively
    # rejected — allow-list is intentional to avoid noise from user-writable
    # areas that were never in the manifest (e.g. logs/, jobs/), but any
    # unexpected .exe / .dll / .pyd / .yaml / .json inside the installed
    # resource root fails the verification.
    unexpected: list = []
    if _schema >= 2:
        _forbidden_ext_when_unlisted = {
            ".exe", ".dll", ".pyd", ".so", ".yaml", ".yml", ".json", ".policy",
        }
        for dirpath, _dirnames, filenames in os.walk(str(root)):
            for fn in filenames:
                full = os.path.join(dirpath, fn)
                if str(Path(full).resolve()) in listed_paths:
                    continue
                # Skip the manifest itself
                if os.path.relpath(full, str(root)).replace(os.sep, "/") == MANIFEST_RELATIVE:
                    continue
                ext = os.path.splitext(fn)[1].lower()
                if ext in _forbidden_ext_when_unlisted:
                    unexpected.append(os.path.relpath(full, str(root)).replace(os.sep, "/"))

    ok = (not malformed) and (not missing) and (not mismatched) and (not unexpected)
    return ok, {
        "ok": bool(ok), "files_checked": len(files),
        "files_ok": ok_count,
        "malformed": malformed,
        "missing": missing,
        "mismatched": mismatched,
        "unexpected": unexpected,
        "manifest_path": str(root / MANIFEST_RELATIVE),
        "installation_root": str(root),
        "manifest_schema_version": _schema,
        "trust_root": manifest.get("trust_root"),
    }
