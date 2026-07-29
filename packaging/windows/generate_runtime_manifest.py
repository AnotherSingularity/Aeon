"""generate_runtime_manifest.py — canonical file-integrity manifest (§W5, §W7).

After PyInstaller builds dist/Aeon, walk every file under the bundle root and
record: relative path (forward slashes), size in bytes, sha256, release id.

The manifest is written to packaging/windows/RUNTIME_MANIFEST.json inside the
bundle. aeon.integrity.verify_installed_manifest reads it at first launch and
before training.

Run on Windows after build:

    python packaging\\windows\\generate_runtime_manifest.py \\
        --bundle dist\\Aeon --release packaging\\windows\\RELEASE.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _sha256(path: str, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--release", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()
    if not bundle.is_dir():
        print(f"[manifest] bundle not a directory: {bundle}", file=sys.stderr)
        return 1
    release_meta = json.load(open(args.release, encoding="utf-8"))

    files_meta = []
    total_bytes = 0
    for root, _, filenames in os.walk(bundle):
        for fn in filenames:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, bundle).replace(os.sep, "/")
            # Skip the manifest itself if it already exists
            if rel == "packaging/windows/RUNTIME_MANIFEST.json":
                continue
            size = os.path.getsize(full)
            total_bytes += size
            files_meta.append({"path": rel, "bytes": size,
                                "sha256": _sha256(full)})
    files_meta.sort(key=lambda x: x["path"])

    manifest = {
        "generated_at": time.time(),
        "release": release_meta,
        "build_architecture": "x64",
        "files": files_meta,
        "total_bytes": total_bytes,
        "file_count": len(files_meta),
    }
    out_path = args.out or str(bundle / "packaging" / "windows" / "RUNTIME_MANIFEST.json")
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, sort_keys=True, indent=2, ensure_ascii=False)
    print(f"[manifest] wrote {out_path} ({len(files_meta)} files, {total_bytes/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
