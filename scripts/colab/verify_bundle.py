"""scripts/colab/verify_bundle.py — verify every bundled file by SHA-256.

Reads SHA256_MANIFEST.json (produced by scripts/colab/build_bundle.py) and
recomputes the SHA-256 of every listed path. Halts non-zero on any drift.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def _sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".",
                    help="bundle root (default: current working directory)")
    ap.add_argument("--manifest", default="SHA256_MANIFEST.json")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    manifest_path = root / args.manifest
    if not manifest_path.exists():
        print(f"ERROR: manifest missing at {manifest_path}", file=sys.stderr)
        return 2
    m = json.loads(manifest_path.read_text(encoding="utf-8"))

    bad, missing, ok = [], [], 0
    for entry in m["files"]:
        rel = entry["path"]
        want = entry["sha256"]
        full = root / rel
        if not full.exists():
            missing.append(rel)
            continue
        got = _sha256(full)
        if got != want:
            bad.append((rel, want, got))
        else:
            ok += 1

    print(json.dumps({
        "ok": len(bad) == 0 and len(missing) == 0,
        "files_ok": ok,
        "files_missing": missing,
        "files_bad": [{"path": r, "expected": w, "got": g}
                       for r, w, g in bad],
        "total_declared": len(m["files"]),
    }, indent=2))
    return 0 if (not bad and not missing) else 3


if __name__ == "__main__":
    sys.exit(main())
