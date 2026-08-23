"""scripts/colab/download_wikitext103.py — fetch, verify, extract WikiText-103 raw.

Canonical URL:
    https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip

Expected byte size: 191,984,949
Expected SHA-256:   91c00ae287f0d699e18605c84afc9e45c192bc6b7797ff8837e5474655a33794

License: CC BY-SA (per Merity et al. WikiText paper).
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path

CANONICAL_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"
EXPECTED_BYTES = 191984949
EXPECTED_SHA256 = "91c00ae287f0d699e18605c84afc9e45c192bc6b7797ff8837e5474655a33794"


def _sha256_file(p: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/content/wikitext-103-raw")
    ap.add_argument("--cache", default="/content/wikitext-103-raw-v1.zip")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cache = Path(args.cache)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download only if the cache is missing or size mismatched.
    need_download = True
    if cache.exists() and cache.stat().st_size == EXPECTED_BYTES:
        need_download = False
    if need_download:
        print(f"[wikitext] downloading {CANONICAL_URL}")
        # -c enables partial-resume; use wget (available on Colab)
        rc = subprocess.call(["wget", "-c", "-q", "--show-progress",
                              "-O", str(cache), CANONICAL_URL])
        if rc != 0 or not cache.exists():
            print(f"ERROR: wget failed rc={rc}", file=sys.stderr)
            return 2

    # Verify byte size FIRST (cheap check).
    got_bytes = cache.stat().st_size
    if got_bytes != EXPECTED_BYTES:
        print(f"ERROR: byte size mismatch: got={got_bytes} want={EXPECTED_BYTES}",
              file=sys.stderr)
        return 3

    # Then SHA-256.
    got_sha = _sha256_file(cache)
    if got_sha != EXPECTED_SHA256:
        print(f"ERROR: sha256 mismatch: got={got_sha} want={EXPECTED_SHA256}",
              file=sys.stderr)
        return 3

    # Extract only if not already present.
    marker = out_dir / "wikitext-103-raw" / "wiki.train.raw"
    if not marker.exists():
        print(f"[wikitext] extracting to {out_dir}")
        with zipfile.ZipFile(cache) as zf:
            zf.extractall(out_dir)
    else:
        print(f"[wikitext] already extracted; skipping")

    train = out_dir / "wikitext-103-raw" / "wiki.train.raw"
    valid = out_dir / "wikitext-103-raw" / "wiki.valid.raw"
    test = out_dir / "wikitext-103-raw" / "wiki.test.raw"
    for p in (train, valid, test):
        if not p.exists():
            print(f"ERROR: extracted file missing: {p}", file=sys.stderr)
            return 4
        print(f"[wikitext] {p.name}: {p.stat().st_size:,} bytes")

    print("[wikitext] OK — byte size and SHA-256 verified, extracted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
