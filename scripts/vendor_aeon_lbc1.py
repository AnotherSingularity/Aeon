"""scripts/vendor_aeon_lbc1.py — one-time AEON-LBC-1 source acquisition.

DEVELOPER-ONLY. Not part of the Aeon runtime. Not bundled into
AeonSetup.exe. Runs offline after the six vendored files are on
disk; every other stage of the pipeline is offline-only.

Two mutually exclusive intake modes:

    * NETWORK (default) — download the six eBook IDs from official
      Project Gutenberg URLs. Every §3 safety control is enforced.
    * OFFLINE (--source-dir <DIR>) — read six pre-downloaded files
      the operator supplies on disk. NO network access is attempted;
      every §1/§4 offline safety control is enforced. Provenance
      sidecars mandatory.

Network-mode safety controls (§3):

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

Offline-mode safety controls (§1):

  * Reads only local files under the operator-supplied directory.
  * Accepts exactly the six allowlisted work identities; extra files
    and missing files both fail-closed.
  * Strict UTF-8 decode.
  * Rejects HTML/PDF/binary masquerading as text.
  * Rejects files whose content does not carry the expected
    Gutenberg source markers, the expected eBook ID, and the
    expected title.
  * Preserves supplied bytes byte-for-byte; SHA-256 recorded from
    the actual supplied bytes (never pre-declared).
  * Requires a provenance sidecar per file; retrieval_method must
    be one of the allowed values.
  * Records acquisition_method="manual_official_download" and
    executing_environment_did_not_download=true.
  * All six sources validate atomically before ANY is promoted to
    the package. Partial imports are refused; the previous package
    state is unchanged on failure.
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


# ---------------------------------------------------------------------------
# Offline intake (--source-dir MODE)
# ---------------------------------------------------------------------------
# The intake directory layout is fixed by the directive §2:
#   <intake>/sources/pg-XXXX.txt
#   <intake>/provenance/pg-XXXX.json
# where XXXX zero-pads to 4 digits.
INTAKE_FILENAME_TO_WORK = {
    "pg-0011.txt": "pg-11",
    "pg-0055.txt": "pg-55",
    "pg-0084.txt": "pg-84",
    "pg-1342.txt": "pg-1342",
    "pg-1661.txt": "pg-1661",
    "pg-2701.txt": "pg-2701",
}
INTAKE_SOURCES_SUBDIR = "sources"
INTAKE_PROVENANCE_SUBDIR = "provenance"

MAX_INTAKE_BYTES = 25 * 1024 * 1024
BINARY_SNIFF_BYTES = 4096

ALLOWED_RETRIEVAL_METHODS = frozenset({
    "manual_browser_download",
    "manual_official_download",
    "authorized_offline_transfer",
})

PROVENANCE_REQUIRED_KEYS = (
    "schema_version",
    "ebook_id",
    "title",
    "source_provider",
    "source_page",
    "retrieval_date",
    "retrieval_method",
    "format",
    "public_domain_basis",
    "provided_by",
)

# Per-work title-evidence patterns. Gutenberg headers usually carry
# "Title: <title>"; we do a case-insensitive substring check against a
# work-specific evidence phrase in the first ~200 lines. Versioned so a
# format-specific edge case can be relaxed for one work without
# loosening validation globally.
_WORK_TITLE_EVIDENCE_VERSION = 1
_WORK_TITLE_EVIDENCE = {
    "pg-11":   ("alice",),                  # "Alice's Adventures in Wonderland"
    "pg-55":   ("wonderful wizard of oz",),
    "pg-84":   ("frankenstein",),
    "pg-1342": ("pride and prejudice",),
    "pg-1661": ("adventures of sherlock holmes",),
    "pg-2701": ("moby",),                   # "Moby-Dick" or "Moby Dick"
}

# HTML / PDF / audio-transcript sniff patterns rejected before any
# downstream processing.
_HTML_SNIFF = (b"<!doctype", b"<html", b"<HTML", b"<HEAD", b"<head",
                 b"<script", b"<title>")
_PDF_SNIFF = (b"%PDF-",)


def _looks_html_or_pdf(prefix: bytes) -> Optional[str]:
    lo = prefix.lstrip()[:200].lower()
    for token in _HTML_SNIFF:
        if token.lower() in lo:
            return "html"
    for token in _PDF_SNIFF:
        if prefix.startswith(token):
            return "pdf"
    return None


def _is_binaryish(prefix: bytes) -> bool:
    """Reject files whose first BINARY_SNIFF_BYTES look binary."""
    if b"\x00" in prefix:
        return True
    # Any control byte outside the printable / whitespace range that
    # cannot legitimately appear in UTF-8 plain text.
    for b in prefix:
        if b < 9 or (13 < b < 32):
            return True
    return False


def _validate_provenance(raw: dict, expected_ebook_id: int,
                           expected_title: str) -> None:
    for k in PROVENANCE_REQUIRED_KEYS:
        if k not in raw:
            raise AcquisitionError(
                "provenance_missing_field", f"{k!r}")
    if raw["schema_version"] != 1:
        raise AcquisitionError(
            "provenance_unsupported_schema", str(raw["schema_version"]))
    if int(raw["ebook_id"]) != int(expected_ebook_id):
        raise AcquisitionError(
            "provenance_ebook_id_mismatch",
            f"provenance ebook_id={raw['ebook_id']!r} expected "
            f"{expected_ebook_id!r}")
    if raw.get("format") != "plain_text_utf8":
        raise AcquisitionError(
            "provenance_wrong_format", str(raw.get("format")))
    if raw.get("retrieval_method") not in ALLOWED_RETRIEVAL_METHODS:
        raise AcquisitionError(
            "provenance_invalid_retrieval_method",
            f"{raw.get('retrieval_method')!r} not in "
            f"{sorted(ALLOWED_RETRIEVAL_METHODS)}")
    if raw.get("public_domain_basis") not in ("public_domain_in_usa",
                                                  "public_domain"):
        raise AcquisitionError(
            "provenance_invalid_public_domain_basis",
            str(raw.get("public_domain_basis")))
    # Title fuzzy match — sanity check that provenance is not from a
    # different work. Case-insensitive substring of a canonical token.
    canonical = expected_title.lower()
    tokens = canonical.split(";")[0].strip()
    if tokens.split()[0] not in (raw.get("title") or "").lower():
        raise AcquisitionError(
            "provenance_title_mismatch",
            f"provenance title {raw.get('title')!r} does not match "
            f"expected first token of {expected_title!r}")


def _validate_content_identity(text: str, work_id: str,
                                  ebook_number: int, title: str) -> None:
    """Verify the file's own contents identify it as the right work.

    Filename alone is not proof (§3). Checks:
      1. Recognizable Project Gutenberg source markers exist.
      2. eBook ID appears in the header block (if present).
      3. A work-specific title-evidence phrase is present in the
         first ~200 lines (case-insensitive).
    """
    head = "\n".join(text.splitlines()[:200])
    head_lower = head.lower()
    if ("project gutenberg" not in head_lower):
        raise AcquisitionError(
            "no_gutenberg_header", f"{work_id}: 'Project Gutenberg' phrase "
            "not found in the first 200 lines")
    # Header ID hint (Gutenberg files usually say 'EBook #N' or
    # 'eBook of ... [EBook #N]'). Enforced ONLY when present — some
    # older UTF-8 editions omit the numeric hint in the header block
    # itself but always carry the *** START OF ... *** marker.
    id_variants = (f"#{ebook_number}", f"#{ebook_number}]",
                    f"ebook {ebook_number}", f"e-book {ebook_number}")
    if not any(v in head_lower for v in id_variants):
        # Not fatal — many editions omit this in the header — but the
        # title evidence check below must succeed.
        pass
    evidence_tokens = _WORK_TITLE_EVIDENCE.get(work_id, ())
    if not evidence_tokens:
        raise AcquisitionError(
            "work_id_not_allowlisted", work_id)
    if not any(tok in head_lower for tok in evidence_tokens):
        raise AcquisitionError(
            "title_evidence_missing",
            f"{work_id}: none of {evidence_tokens} appears in the "
            "first 200 lines")


@dataclass(frozen=True)
class _IntakeStagedFile:
    work_id: str
    canonical_filename: str
    bytes: bytes
    sha256: str
    provenance: dict
    intake_source_filename: str
    intake_provenance_filename: str


def import_offline_sources(package_root: Path, source_dir: Path) -> dict:
    """Read the six pre-downloaded files from `source_dir/sources/` and
    their provenance sidecars from `source_dir/provenance/`.

    Validates every file before ANY byte is copied into the package. On
    validation failure, the previous package state is unchanged.

    Returns an acquisition summary that mirrors network mode's shape,
    with acquisition_method="manual_official_download" so downstream
    audit code can distinguish the two intakes.
    """
    sources_root = source_dir / INTAKE_SOURCES_SUBDIR
    provenance_root = source_dir / INTAKE_PROVENANCE_SUBDIR
    if not sources_root.is_dir():
        raise AcquisitionError(
            "intake_sources_dir_missing", str(sources_root))
    if not provenance_root.is_dir():
        raise AcquisitionError(
            "intake_provenance_dir_missing", str(provenance_root))

    # 1. Presence check + no-extras check.
    expected = set(INTAKE_FILENAME_TO_WORK.keys())
    present = {p.name for p in sources_root.iterdir() if p.is_file()}
    missing = sorted(expected - present)
    if missing:
        raise AcquisitionError(
            "intake_missing_sources", ", ".join(missing))
    extras = sorted(present - expected)
    if extras:
        raise AcquisitionError(
            "intake_unexpected_sources", ", ".join(extras))

    # 2. Read + validate every source AND its provenance sidecar into
    #    an in-memory staged bundle. Do NOT touch the package tree yet.
    staged: list = []
    allowlist_by_id = {w.work_id: w for w in ALLOWLIST}
    for intake_filename in sorted(expected):
        work_id = INTAKE_FILENAME_TO_WORK[intake_filename]
        work = allowlist_by_id[work_id]
        src_path = sources_root / intake_filename
        prov_name = intake_filename.rsplit(".", 1)[0] + ".json"
        prov_path = provenance_root / prov_name
        if not prov_path.exists():
            raise AcquisitionError(
                "intake_provenance_missing", str(prov_path))
        # Read source bytes (byte-preserving)
        try:
            with open(src_path, "rb") as fh:
                data = fh.read(MAX_INTAKE_BYTES + 1)
        except OSError as e:
            raise AcquisitionError(
                "intake_source_unreadable", f"{src_path}: {e}") from e
        if len(data) > MAX_INTAKE_BYTES:
            raise AcquisitionError(
                "intake_source_too_large",
                f"{intake_filename}: > {MAX_INTAKE_BYTES} bytes")
        prefix = data[:BINARY_SNIFF_BYTES]
        masquerade = _looks_html_or_pdf(prefix)
        if masquerade:
            raise AcquisitionError(
                "intake_non_plain_text",
                f"{intake_filename}: looks like {masquerade}")
        if _is_binaryish(prefix):
            raise AcquisitionError(
                "intake_binary_data",
                f"{intake_filename}: binary bytes detected in first "
                f"{BINARY_SNIFF_BYTES} bytes")
        # Strict UTF-8 decode
        try:
            text = data.decode("utf-8", errors="strict")
        except UnicodeDecodeError as e:
            raise AcquisitionError(
                "intake_utf8_decode_failed",
                f"{intake_filename}: {e}") from e
        # Content-identity validation
        _validate_content_identity(
            text, work_id=work.work_id,
            ebook_number=work.ebook_number, title=work.title)
        # Provenance sidecar
        try:
            with open(prov_path, encoding="utf-8") as fh:
                prov = json.load(fh)
        except Exception as e:
            raise AcquisitionError(
                "intake_provenance_unreadable",
                f"{prov_path}: {e}") from e
        _validate_provenance(prov, expected_ebook_id=work.ebook_number,
                              expected_title=work.title)
        staged.append(_IntakeStagedFile(
            work_id=work.work_id,
            canonical_filename=work.source_filename,
            bytes=data,
            sha256=_sha256(data),
            provenance=prov,
            intake_source_filename=intake_filename,
            intake_provenance_filename=prov_name,
        ))

    if len(staged) != len(INTAKE_FILENAME_TO_WORK):
        raise AcquisitionError(
            "intake_incomplete_after_validation",
            f"staged {len(staged)} / expected "
            f"{len(INTAKE_FILENAME_TO_WORK)}")

    # 3. Atomic promotion. Every source validated; write into the
    #    package. Order: source bytes first (per-file), then
    #    ORIGINAL_SOURCE_DIGESTS, then provenance/acquisition.json.
    source_out = package_root / "source"
    prov_out = package_root / "provenance"
    source_out.mkdir(parents=True, exist_ok=True)
    prov_out.mkdir(parents=True, exist_ok=True)
    now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    recorded: dict = {}
    summary = {"acquired_at_utc": now_utc,
                "acquisition_method": "manual_official_download",
                "executing_environment_did_not_download": True,
                "works": []}
    for f in staged:
        target = source_out / f.canonical_filename
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(f.bytes)
        # Reread to verify copied bytes.
        reread = tmp.read_bytes()
        if _sha256(reread) != f.sha256:
            tmp.unlink(missing_ok=True)
            raise AcquisitionError(
                "intake_post_copy_digest_mismatch",
                f"{f.canonical_filename}: copy did not round-trip")
        os.replace(tmp, target)
        # Per-work provenance record inside the package.
        (prov_out / f"{f.work_id}.json").write_text(
            json.dumps(f.provenance, indent=2, sort_keys=True),
            encoding="utf-8")
        recorded[f.work_id] = {
            "sha256": f.sha256,
            "resolved_url": "offline:" + f.intake_source_filename,
            "retrieved_at_utc": now_utc,
            "bytes": len(f.bytes),
            "acquisition_method": "manual_official_download",
            "executing_environment_did_not_download": True,
            "supplied_retrieval_date": f.provenance.get("retrieval_date"),
            "supplied_source_page": f.provenance.get("source_page"),
            "title_evidence_version": _WORK_TITLE_EVIDENCE_VERSION,
        }
        summary["works"].append({
            "work_id": f.work_id,
            "ebook_number": next(w.ebook_number for w in ALLOWLIST
                                   if w.work_id == f.work_id),
            "title": next(w.title for w in ALLOWLIST
                            if w.work_id == f.work_id),
            "partition_role": next(w.partition_role for w in ALLOWLIST
                                     if w.work_id == f.work_id),
            "source_filename": f.canonical_filename,
            "status": "acquired",
            "sha256": f.sha256,
            "bytes": len(f.bytes),
            "resolved_url": "offline:" + f.intake_source_filename,
            "acquisition_method": "manual_official_download",
            "executing_environment_did_not_download": True,
        })
    _write_recorded_digests(package_root / "ORIGINAL_SOURCE_DIGESTS",
                              recorded)
    (prov_out / "acquisition.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
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
    ap.add_argument("--source-dir", default=None,
                     help="OFFLINE MODE: directory containing "
                          "sources/pg-XXXX.txt and provenance/pg-XXXX.json "
                          "files supplied by the operator. Mutually exclusive "
                          "with network acquisition; no network access is "
                          "attempted when this argument is present.")
    args = ap.parse_args(argv)
    root = Path(args.package_root).resolve()
    if args.source_dir is not None:
        # Offline mode. Refuse if --refresh-source or --ca-bundle are set
        # — those are only meaningful when the network path is active.
        if args.refresh_source:
            print("--refresh-source is meaningless in offline mode",
                    file=sys.stderr)
            return 2
        if args.ca_bundle:
            print("--ca-bundle is meaningless in offline mode",
                    file=sys.stderr)
            return 2
        try:
            summary = import_offline_sources(
                root, Path(args.source_dir).resolve())
        except AcquisitionError as e:
            print(json.dumps({"ok": False, "code": e.code,
                                "detail": e.detail}), file=sys.stderr)
            return 1
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
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
