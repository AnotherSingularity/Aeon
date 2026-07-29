"""release_metadata.py — build packaging/windows/RELEASE.json for the current build.

Reads git for the source_commit, folds in the pinned semantic_version, and
emits a JSON that aeon.version.RELEASE_METADATA will consume at runtime.

Never records signing keys, certificate passwords, or any other secret.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT = {
    "product_name": "Aeon",
    "file_description": "Aeon defensive-resilience runtime",
    "semantic_version": "0.2.3",
    "architecture": "x64",
    "copyright": "Aeon contributors",
    "publisher": "Aeon",
}


def _git_head() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                            text=True, timeout=5,
                            cwd=str(Path(__file__).resolve().parent.parent.parent))
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--build-type", choices=["development", "release"],
                     default="development")
    ap.add_argument("--signed", action="store_true",
                     help="mark the build as signed (only after SignTool succeeds)")
    ap.add_argument("--semantic-version", default=DEFAULT["semantic_version"])
    args = ap.parse_args()

    payload = dict(DEFAULT)
    payload["semantic_version"] = args.semantic_version
    payload["build_type"] = args.build_type
    payload["source_commit"] = _git_head()
    payload["signed"] = bool(args.signed)
    payload["publisher"] = ("Aeon" if args.signed else "Aeon (development)")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, sort_keys=True, indent=2, ensure_ascii=False)
    print(f"[release_metadata] wrote {args.out}: "
          f"v{payload['semantic_version']}  {payload['source_commit'][:8]}  "
          f"build_type={payload['build_type']}  signed={payload['signed']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
