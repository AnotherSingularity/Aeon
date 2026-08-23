"""scripts/colab/download_wikitext103.py — fetch, verify, extract WikiText-103 raw.

Legacy canonical URL (kept for provenance):
    https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
    (Salesforce/MetaMind published the file here; the S3 bucket now
    returns 301/403 in some Colab regions — see EN-COLAB-C notes.)

Immutable mirror used at download time (revision-pinned):
    https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/
        3555105b17ae31cc619a136fac72dbe2865c3738/wikitext-103-raw-v1.zip?download=true

Both URLs deliver the SAME bytes; identity is enforced by BOTH the byte-
count check AND the SHA-256 check BEFORE extraction. The mirror URL is
git-revision-pinned (Hugging Face `resolve/<sha>/...`) so it cannot
silently drift. TLS verification is never disabled.

Expected byte size: 191,984,949
Expected SHA-256:   91c00ae287f0d699e18605c84afc9e45c192bc6b7797ff8837e5474655a33794

License: CC BY-SA (per Merity et al. WikiText paper).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

# NOTE: URLs kept as single unbroken string literals so static tests
# and audit tools can grep them without whitespace surprises.
LEGACY_CANONICAL_URL = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip"

IMMUTABLE_MIRROR_URL = "https://huggingface.co/datasets/mattdangerw/wikitext-103-raw/resolve/3555105b17ae31cc619a136fac72dbe2865c3738/wikitext-103-raw-v1.zip?download=true"

# Ordered download attempts. Mirror is tried FIRST because the legacy
# S3 URL now returns 301/403 in some Colab regions. Any URL that
# delivers bytes with the correct size + SHA-256 satisfies the download.
DOWNLOAD_URLS = (
    ("hf_mirror_revision_pinned", IMMUTABLE_MIRROR_URL),
    ("legacy_s3_canonical",       LEGACY_CANONICAL_URL),
)

EXPECTED_BYTES = 191984949
EXPECTED_SHA256 = "91c00ae287f0d699e18605c84afc9e45c192bc6b7797ff8837e5474655a33794"


def _sha256_file(p: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return h.hexdigest()


def _download_one(url: str, dest: Path) -> int:
    """Download url to dest with wget (TLS verify ON). Returns rc."""
    # -c enables partial-resume; --tries=3 for transient blips.
    # TLS certificate verification is always ENABLED — we never pass
    # any option that would skip cert validation.
    return subprocess.call(["wget", "--tries=3", "--continue",
                             "-q", "--show-progress", "-O", str(dest), url])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/content/wikitext-103-raw")
    ap.add_argument("--cache", default="/content/wikitext-103-raw-v1.zip")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    cache = Path(args.cache)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reuse an existing cache only if it already matches the expected
    # size — never trust a truncated/partial file.
    if cache.exists() and cache.stat().st_size == EXPECTED_BYTES:
        print(f"[wikitext] using cached {cache} ({EXPECTED_BYTES:,} bytes)")
    else:
        if cache.exists():
            print(f"[wikitext] removing wrong-size cache: "
                  f"{cache.stat().st_size} != {EXPECTED_BYTES}")
            cache.unlink()

        errors = []
        for label, url in DOWNLOAD_URLS:
            print(f"[wikitext] trying {label}: {url}")
            rc = _download_one(url, cache)
            if rc != 0 or not cache.exists():
                errors.append(f"{label}: wget rc={rc}")
                if cache.exists():
                    cache.unlink()
                continue
            got = cache.stat().st_size
            if got != EXPECTED_BYTES:
                errors.append(f"{label}: byte size {got} != {EXPECTED_BYTES}")
                cache.unlink()
                continue
            print(f"[wikitext] {label} delivered {got:,} bytes (matches expected)")
            break
        else:
            print("ERROR: every download URL failed:", file=sys.stderr)
            for e in errors: print(f"  {e}", file=sys.stderr)
            return 2

    # Byte size (cheap check; already verified above but re-assert for
    # any concurrent-tool safety).
    got_bytes = cache.stat().st_size
    if got_bytes != EXPECTED_BYTES:
        print(f"ERROR: byte size mismatch after download: got={got_bytes} "
              f"want={EXPECTED_BYTES}", file=sys.stderr)
        return 3

    # SHA-256 — the authoritative identity check.
    got_sha = _sha256_file(cache)
    if got_sha != EXPECTED_SHA256:
        print(f"ERROR: sha256 mismatch: got={got_sha} want={EXPECTED_SHA256}",
              file=sys.stderr)
        return 3
    print(f"[wikitext] sha256 verified: {got_sha}")

    # Extract only if not already present.
    marker = out_dir / "wikitext-103-raw" / "wiki.train.raw"
    if not marker.exists():
        print(f"[wikitext] extracting to {out_dir}")
        with zipfile.ZipFile(cache) as zf:
            zf.extractall(out_dir)
    else:
        print(f"[wikitext] already extracted; skipping")

    for name in ("wiki.train.raw", "wiki.valid.raw", "wiki.test.raw"):
        p = out_dir / "wikitext-103-raw" / name
        if not p.exists():
            print(f"ERROR: extracted file missing: {p}", file=sys.stderr)
            return 4
        print(f"[wikitext] {p.name}: {p.stat().st_size:,} bytes")

    # Record provenance for the operator.
    prov = out_dir / "download_provenance.json"
    prov.write_text(json.dumps({
        "legacy_canonical_url": LEGACY_CANONICAL_URL,
        "immutable_mirror_url": IMMUTABLE_MIRROR_URL,
        "expected_byte_size": EXPECTED_BYTES,
        "expected_sha256": EXPECTED_SHA256,
        "actual_byte_size": got_bytes,
        "actual_sha256": got_sha,
        "match": got_sha == EXPECTED_SHA256 and got_bytes == EXPECTED_BYTES,
        "tls_verification": "ENABLED",
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[wikitext] OK — byte size and SHA-256 verified, extracted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
