"""scripts/vendor_aeon_lbc1.py — one-time AEON-LBC-1 source acquisition.

DEVELOPER-ONLY. Not part of the Aeon runtime. Not bundled into
AeonSetup.exe. Runs offline after the six vendored files are on
disk; every other stage of the pipeline is offline-only.

Safety controls (§3):

  * Exactly six eBook IDs allowlisted; nothing else is fetched.
  * Official Gutenberg UTF-8 plain-text URLs only.
  * Redirects outside gutenberg.org / *.gutenberg.org are refused.
  * HTTPS enforced; HTTP responses rejected.
  * Response size ceiling (25 MiB per file); connection timeout.
  * Content-Type must include text/plain; HTML responses rejected.
  * Retrieval timestamp (UTC) and resolved source URL recorded.
  * SHA-256 computed before any preprocessing.
  * Complete original file preserved byte-for-byte under source/.
  * Never executes downloaded content.
  * Never imports downloaded content as Python.
  * Never reads any Aeon repository file during acquisition (an
    accidental corpus contamination guard).
  * Refuses to overwrite an existing source file whose digest differs
    from the recorded digest unless --refresh-source is explicitly set.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Allowlist — exactly six IDs, no substitutions
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GutenbergWork:
    work_id: str
    ebook_number: int
    title: str
    partition_role: str  # "train" | "calibration" | "validation" | "test"
    source_filename: str
    primary_url: str
    fallback_urls: tuple


ALLOWLIST: tuple = (
    GutenbergWork(
        work_id="pg-2701",
        ebook_number=2701,
        title="Moby-Dick; Or, The Whale",
        partition_role="train",
        source_filename="pg-2701-moby-dick.txt",
        primary_url="https://www.gutenberg.org/cache/epub/2701/pg2701.txt",
        fallback_urls=(
            "https://www.gutenberg.org/ebooks/2701.txt.utf-8",
        ),
    ),
    GutenbergWork(
        work_id="pg-1342",
        ebook_number=1342,
        title="Pride and Prejudice",
        partition_role="train",
        source_filename="pg-1342-pride-and-prejudice.txt",
        primary_url="https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
        fallback_urls=(
            "https://www.gutenberg.org/ebooks/1342.txt.utf-8",
        ),
    ),
    GutenbergWork(
        work_id="pg-11",
        ebook_number=11,
        title="Alice's Adventures in Wonderland",
        partition_role="train",
        source_filename="pg-0011-alice.txt",
        primary_url="https://www.gutenberg.org/cache/epub/11/pg11.txt",
        fallback_urls=(
            "https://www.gutenberg.org/ebooks/11.txt.utf-8",
        ),
    ),
    GutenbergWork(
        work_id="pg-84",
        ebook_number=84,
        title="Frankenstein; Or, The Modern Prometheus",
        partition_role="calibration",
        source_filename="pg-0084-frankenstein.txt",
        primary_url="https://www.gutenberg.org/cache/epub/84/pg84.txt",
        fallback_urls=(
            "https://www.gutenberg.org/ebooks/84.txt.utf-8",
        ),
    ),
    GutenbergWork(
        work_id="pg-55",
        ebook_number=55,
        title="The Wonderful Wizard of Oz",
        partition_role="validation",
        source_filename="pg-0055-wizard-of-oz.txt",
        primary_url="https://www.gutenberg.org/cache/epub/55/pg55.txt",
        fallback_urls=(
            "https://www.gutenberg.org/ebooks/55.txt.utf-8",
        ),
    ),
    GutenbergWork(
        work_id="pg-1661",
        ebook_number=1661,
        title="The Adventures of Sherlock Holmes",
        partition_role="test",
        source_filename="pg-1661-sherlock-holmes.txt",
        primary_url="https://www.gutenberg.org/cache/epub/1661/pg1661.txt",
        fallback_urls=(
            "https://www.gutenberg.org/ebooks/1661.txt.utf-8",
        ),
    ),
)

MAX_RESPONSE_BYTES = 25 * 1024 * 1024  # 25 MiB — every allowlisted work is smaller
CONNECT_TIMEOUT_SECONDS = 30.0
ALLOWED_HOST_SUFFIXES = ("gutenberg.org",)


class AcquisitionError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# Safety helpers
# ---------------------------------------------------------------------------
def _assert_https(url: str) -> None:
    if not url.startswith("https://"):
        raise AcquisitionError("http_forbidden", url)


def _assert_allowed_host(url: str) -> None:
    parsed = urllib.parse.urlparse(url)
    host = (parsed.hostname or "").lower()
    if not any(host == h or host.endswith("." + h)
                 for h in ALLOWED_HOST_SUFFIXES):
        raise AcquisitionError("host_not_allowlisted", host)


def _reject_html(headers) -> None:
    ctype = (headers.get("Content-Type") or "").lower()
    if "text/plain" not in ctype:
        raise AcquisitionError("non_text_plain_response", ctype)


class _NoOffsiteRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse any redirect that would move off gutenberg.org."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _assert_https(newurl)
        _assert_allowed_host(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _fetch_bytes(url: str, *, ca_bundle_path: Optional[str] = None) -> bytes:
    _assert_https(url)
    _assert_allowed_host(url)
    ctx = ssl.create_default_context(cafile=ca_bundle_path)
    opener = urllib.request.build_opener(
        _NoOffsiteRedirect(),
        urllib.request.HTTPSHandler(context=ctx),
    )
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "aeon-lbc1-vendor/1 (research; offline-cache)"},
    )
    with opener.open(req, timeout=CONNECT_TIMEOUT_SECONDS) as resp:
        if resp.status != 200:
            raise AcquisitionError("http_not_200", str(resp.status))
        _reject_html(resp.headers)
        buf = bytearray()
        for chunk in iter(lambda: resp.read(1 << 16), b""):
            buf.extend(chunk)
            if len(buf) > MAX_RESPONSE_BYTES:
                raise AcquisitionError(
                    "response_too_large",
                    f"> {MAX_RESPONSE_BYTES} bytes")
        return bytes(buf)


def _sha256(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


# ---------------------------------------------------------------------------
# Vendor
# ---------------------------------------------------------------------------
def _load_recorded_digests(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_recorded_digests(path: Path, digests: dict) -> None:
    path.write_text(json.dumps(digests, indent=2, sort_keys=True),
                     encoding="utf-8")


def vendor_all(package_root: Path, *, refresh_source: bool = False,
                 ca_bundle_path: Optional[str] = None) -> dict:
    """Acquire (or verify) every allowlisted work. Returns a summary
    dict; the caller writes ORIGINAL_SOURCE_DIGESTS + acquisition.json."""
    source_dir = package_root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    digests_path = package_root / "ORIGINAL_SOURCE_DIGESTS"
    recorded = _load_recorded_digests(digests_path)
    summary = {"acquired_at_utc": time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "works": []}

    for work in ALLOWLIST:
        target = source_dir / work.source_filename
        row = {"work_id": work.work_id,
                "ebook_number": work.ebook_number,
                "title": work.title,
                "partition_role": work.partition_role,
                "source_filename": work.source_filename}
        if target.exists() and not refresh_source:
            data = target.read_bytes()
            digest = _sha256(data)
            recorded_digest = recorded.get(work.work_id, {}).get("sha256")
            if recorded_digest is None:
                raise AcquisitionError(
                    "digest_conflict",
                    f"{work.work_id}: on-disk file has no recorded "
                    f"digest — refuse to accept as authoritative. Pass "
                    "--refresh-source to acquire fresh.")
            if recorded_digest != digest:
                raise AcquisitionError(
                    "digest_conflict",
                    f"{work.work_id}: recorded={recorded_digest} vs "
                    f"on-disk={digest}")
            row.update({"status": "already_present",
                         "sha256": digest,
                         "bytes": len(data),
                         "resolved_url": recorded[work.work_id].get(
                             "resolved_url", "cached")})
            summary["works"].append(row)
            continue

        urls = (work.primary_url,) + work.fallback_urls
        last_error = None
        data = None
        resolved_url = None
        for u in urls:
            try:
                data = _fetch_bytes(u, ca_bundle_path=ca_bundle_path)
                resolved_url = u
                break
            except (AcquisitionError, urllib.error.URLError, TimeoutError,
                     ssl.SSLError, OSError) as e:
                last_error = e
                continue
        if data is None:
            row.update({"status": "acquisition_failed",
                         "error": repr(last_error)})
            summary["works"].append(row)
            continue

        digest = _sha256(data)
        if target.exists() and not refresh_source:
            existing = _sha256(target.read_bytes())
            if existing != digest:
                raise AcquisitionError(
                    "digest_conflict",
                    f"{work.work_id}: {existing} vs new {digest} — pass "
                    "--refresh-source explicitly")
        target.write_bytes(data)
        recorded[work.work_id] = {"sha256": digest,
                                    "resolved_url": resolved_url,
                                    "retrieved_at_utc": summary["acquired_at_utc"],
                                    "bytes": len(data)}
        row.update({"status": "acquired",
                     "sha256": digest,
                     "bytes": len(data),
                     "resolved_url": resolved_url})
        summary["works"].append(row)
    _write_recorded_digests(digests_path, recorded)
    (package_root / "provenance").mkdir(exist_ok=True)
    (package_root / "provenance" / "acquisition.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package-root",
                     default="research-data/AEON-LBC-1",
                     help="Local package root (default: %(default)s)")
    ap.add_argument("--refresh-source", action="store_true",
                     help="Overwrite existing source files even when digest "
                          "differs (dangerous — invalidates every downstream "
                          "step). Default: off.")
    ap.add_argument("--ca-bundle", default=os.environ.get("REQUESTS_CA_BUNDLE"))
    args = ap.parse_args(argv)
    root = Path(args.package_root).resolve()
    summary = vendor_all(root, refresh_source=args.refresh_source,
                          ca_bundle_path=args.ca_bundle)
    all_ok = all(r["status"] in ("acquired", "already_present")
                   for r in summary["works"])
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not all_ok:
        print("acquisition_incomplete", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
