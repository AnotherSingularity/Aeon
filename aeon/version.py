"""aeon/version.py — release metadata read from packaging/windows/RELEASE.json when
present (produced by the Windows build), else source-tree defaults."""
from __future__ import annotations

import json
import os
from pathlib import Path

from aeon.windows_paths import installed_resource_root

SOURCE_DEFAULT = {
    "product_name": "Aeon",
    "file_description": "Aeon defensive-resilience runtime",
    "semantic_version": "0.2.3",
    "source_commit": "unknown",
    "architecture": "x64",
    "copyright": "Aeon contributors",
    "publisher": "Aeon (development)",
    "build_type": "development",
    "signed": False,
}


def _load() -> dict:
    for candidate in ("packaging/windows/RELEASE.json", "RELEASE.json"):
        path = installed_resource_root() / candidate
        if path.exists():
            try:
                data = json.load(open(path, encoding="utf-8"))
                merged = dict(SOURCE_DEFAULT)
                merged.update({k: v for k, v in data.items() if k in SOURCE_DEFAULT
                                or k in ("signed", "sha256_manifest")})
                return merged
            except Exception:
                pass
    # Best-effort source_commit from git
    try:
        import subprocess
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=2,
                              cwd=str(installed_resource_root())).stdout.strip()
        if rev:
            out = dict(SOURCE_DEFAULT); out["source_commit"] = rev
            return out
    except Exception:
        pass
    return dict(SOURCE_DEFAULT)


RELEASE_METADATA = _load()
