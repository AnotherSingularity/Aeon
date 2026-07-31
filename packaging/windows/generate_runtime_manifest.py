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

    # PyInstaller 6.x onedir layout: <bundle>/Aeon.exe + <bundle>/_internal/*
    # `_internal/` is the directory `sys._MEIPASS` points at at runtime, which
    # `aeon.windows_paths.installed_resource_root()` returns and
    # `aeon.integrity.verify_installed_manifest()` resolves against. Every
    # immutable runtime file lives under `_internal/`, so the manifest must
    # (a) enumerate paths RELATIVE to `_internal/`, and (b) live at
    # `_internal/packaging/windows/RUNTIME_MANIFEST.json`. On older PyInstaller
    # or unit-test scaffolding without `_internal/`, we fall back to bundle-root.
    internal = bundle / "_internal"
    if internal.is_dir():
        walk_root = internal
        resource_root_relative = "_internal"
    else:
        walk_root = bundle
        resource_root_relative = "."

    files_meta = []
    total_bytes = 0
    # W10-6: also enumerate top-level files (Aeon.exe, launcher shims, etc.)
    # so the runtime manifest covers them too. They are recorded with a
    # "../" prefix relative to the resource_root (`_internal/`) so the
    # verifier can distinguish top-level from internal at load time. The
    # audit's A9 finding was that a modification to top-level Aeon.exe
    # was not detected by verify_installed_manifest — that stops here.
    top_level_files = []
    if internal.is_dir():
        for name in sorted(os.listdir(bundle)):
            full = bundle / name
            if full.is_file():
                rel = f"../{name}"
                size = full.stat().st_size
                total_bytes += size
                top_level_files.append({"path": rel, "bytes": size,
                                          "sha256": _sha256(str(full)),
                                          "scope": "top_level"})
    files_meta.extend(top_level_files)

    for root, _, filenames in os.walk(walk_root):
        for fn in filenames:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, walk_root).replace(os.sep, "/")
            # Skip the manifest itself (chicken-and-egg) and its SHA-256
            # sidecar (W10-7/A13: the sidecar hashes the manifest for the
            # Inno pre-install check; listing itself inside the manifest
            # would be a circular reference).
            if rel == "packaging/windows/RUNTIME_MANIFEST.json":
                continue
            if rel == "packaging/windows/RUNTIME_MANIFEST.sha256":
                continue
            size = os.path.getsize(full)
            total_bytes += size
            files_meta.append({"path": rel, "bytes": size,
                                "sha256": _sha256(full),
                                "scope": "internal"})
    files_meta.sort(key=lambda x: x["path"])

    manifest = {
        "manifest_schema_version": 2,  # bumped from implicit 1 by W10-6
        "generated_at": time.time(),
        "release": release_meta,
        "build_architecture": "x64",
        "resource_root_relative_to_bundle": resource_root_relative,
        "trust_root": {
            # W10-6: honest trust-posture record. The current unsigned
            # development build authenticates the manifest against
            # accidental corruption via SHA-256 per file. It does NOT
            # authenticate against an adversary who can replace both the
            # manifest and every file it lists. The signed-manifest /
            # embedded-digest trust root arrives when Authenticode
            # signing lands in the Tier A build.
            "kind": "sha256_per_file",
            "signed_manifest": False,
            "adversary_integrity_scope": "none",
            "accidental_integrity_scope": "full_bundle_including_top_level",
        },
        "files": files_meta,
        "total_bytes": total_bytes,
        "file_count": len(files_meta),
    }
    out_path = args.out or str(walk_root / "packaging" / "windows" / "RUNTIME_MANIFEST.json")
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, sort_keys=True, indent=2, ensure_ascii=False)
    # W10-7/A13: emit a SHA-256 sidecar so the Inno Setup pre-install check
    # can verify the manifest payload byte-for-byte before install (not just
    # its presence). The manifest itself already hashes every file it lists,
    # so a matching manifest digest transitively verifies the whole payload.
    sidecar_path = str(Path(out_path).with_suffix(".sha256"))
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        fh.write(_sha256(out_path))
    print(f"[manifest] wrote {out_path} ({len(files_meta)} files, {total_bytes/1e6:.1f} MB)")
    print(f"[manifest] wrote {sidecar_path} (sha256 sidecar for installer pre-install check)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
