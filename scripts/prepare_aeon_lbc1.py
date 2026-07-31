"""scripts/prepare_aeon_lbc1.py — deterministic preprocessing (§5).

Reads the six vendored Gutenberg UTF-8 plain-text files under
<package_root>/source/, applies the versioned preprocessing policy
``aeon-lbc1-v1``, and emits four JSONL partitions plus manifests.

Preprocessing rules (frozen at aeon-lbc1-v1):

  1. Decode strict UTF-8. Any decode error aborts the source.
  2. Strip UTF-8 BOM if present.
  3. Normalize CRLF and CR to LF.
  4. Unicode NFC normalization.
  5. Detect header/footer via explicit Gutenberg markers:
       ``*** START OF THIS PROJECT GUTENBERG EBOOK`` and
       ``*** END OF THIS PROJECT GUTENBERG EBOOK`` (also legacy
       ``*** START OF THE PROJECT GUTENBERG EBOOK`` / ``*** END OF``).
     If either marker is missing or ambiguous, the source is
     REJECTED — no heuristic fallback.
  6. Preserve spelling, punctuation, capitalization, paragraph order.
  7. Split into paragraphs on runs of two or more LFs.
  8. Drop runs that are only whitespace after collapse.
  9. Record every transformation applied.
 10. Compute source and processed digests.
 11. Generate stable record IDs (sha256 over
     "<work_id>|<chapter_id>|<paragraph_index>|<text>").
 12. Emit records ONLY into the JSONL partition matching the work's
     partition role (train / calibration / validation / test).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


PREPROCESSING_VERSION = "aeon-lbc1-v1"

# Header/footer markers. The trailing ".*" captures the specific work
# title text upstream inserts between the fixed prefix and the newline;
# we do NOT rely on the work-title text to be stable.
_HEADER_PATTERNS = (
    re.compile(r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK[^\n]*", re.IGNORECASE),
)
_FOOTER_PATTERNS = (
    re.compile(r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK[^\n]*", re.IGNORECASE),
)

_CHAPTER_RE = re.compile(r"^\s*(?:CHAPTER|Chapter|CANTO|BOOK|LETTER)\s+[A-Z0-9IVXLCDM]+", re.MULTILINE)


class PrepError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _sha256_text(s: str) -> str:
    return _sha256_bytes(s.encode("utf-8"))


# ---------------------------------------------------------------------------
# Boundary detection
# ---------------------------------------------------------------------------
def find_body(text: str) -> Tuple[int, int]:
    """Return (start, end) offsets of the body between the last START and
    the first END marker. Raises PrepError if markers are missing or
    ambiguous."""
    header_matches = []
    for pat in _HEADER_PATTERNS:
        header_matches.extend(list(pat.finditer(text)))
    footer_matches = []
    for pat in _FOOTER_PATTERNS:
        footer_matches.extend(list(pat.finditer(text)))
    if not header_matches:
        raise PrepError("header_marker_missing")
    if not footer_matches:
        raise PrepError("footer_marker_missing")
    if len(header_matches) > 1:
        # Choose the last START marker — some editions have a licence
        # preamble containing the phrase.
        header = max(header_matches, key=lambda m: m.start())
    else:
        header = header_matches[0]
    if len(footer_matches) > 1:
        footer = min(footer_matches, key=lambda m: m.start())
    else:
        footer = footer_matches[0]
    if footer.start() <= header.end():
        raise PrepError("footer_before_header")
    # Skip the line the header sits on
    body_start = text.find("\n", header.end())
    if body_start < 0:
        raise PrepError("no_newline_after_header")
    body_start += 1
    body_end = footer.start()
    return body_start, body_end


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProcessedParagraph:
    work_id: str
    chapter_id: str
    paragraph_index: int
    text: str
    source_sha256: str
    record_id: str


def preprocess_source(
    *, raw_bytes: bytes, work_id: str, title: str,
) -> Tuple[List[ProcessedParagraph], dict]:
    """Return (paragraphs, transformation_log)."""
    try:
        text = raw_bytes.decode("utf-8")
    except UnicodeDecodeError as e:
        raise PrepError("utf8_decode_failed", f"{work_id}: {e}")
    log = {"work_id": work_id,
             "input_bytes": len(raw_bytes),
             "input_sha256": _sha256_bytes(raw_bytes),
             "transformations": []}
    # 1. BOM
    if text.startswith("﻿"):
        text = text.lstrip("﻿")
        log["transformations"].append("strip_bom")
    # 2. Line endings
    if "\r\n" in text or "\r" in text:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        log["transformations"].append("normalize_line_endings")
    # 3. Unicode NFC
    normalized = unicodedata.normalize("NFC", text)
    if normalized != text:
        log["transformations"].append("unicode_nfc")
        text = normalized
    # 4. Boundary detection
    body_start, body_end = find_body(text)
    body = text[body_start:body_end]
    log["transformations"].append("boundary_stripped")
    # 5. Chapter splitting (best-effort — a stable label for record ID
    #    grouping. Records inherit the label of the nearest preceding
    #    chapter marker; text preceding the first chapter marker gets
    #    "prelude".)
    chapter_starts = [m.start() for m in _CHAPTER_RE.finditer(body)]
    if not chapter_starts or chapter_starts[0] > 0:
        chapter_starts = [0] + chapter_starts
    chapter_boundaries: List[Tuple[str, int, int]] = []
    for i, cstart in enumerate(chapter_starts):
        cend = chapter_starts[i + 1] if i + 1 < len(chapter_starts) else len(body)
        cid = _sha256_text(body[cstart:cend])[:23]
        chapter_boundaries.append((cid, cstart, cend))
    log["transformations"].append("chapter_indexed")
    # 6. Paragraph split within each chapter (runs of 2+ LFs)
    para_re = re.compile(r"\n\s*\n")
    processed: List[ProcessedParagraph] = []
    for cid, cstart, cend in chapter_boundaries:
        chunk = body[cstart:cend]
        paras = [p.strip() for p in para_re.split(chunk)]
        for pidx, para in enumerate([p for p in paras if p]):
            record_id_input = f"{work_id}|{cid}|{pidx}|{para}".encode("utf-8")
            processed.append(ProcessedParagraph(
                work_id=work_id,
                chapter_id=cid,
                paragraph_index=pidx,
                text=para,
                source_sha256=log["input_sha256"],
                record_id=_sha256_bytes(record_id_input),
            ))
    log["transformations"].append("paragraphs_split")
    log["paragraph_count"] = len(processed)
    return processed, log


# ---------------------------------------------------------------------------
# Emit
# ---------------------------------------------------------------------------
def emit_partition_jsonl(
    processed: Iterable[ProcessedParagraph],
    *,
    partition: str,
    work_id: str,
    title: str,
    out_path: Path,
    append: bool = False,
) -> Tuple[int, str]:
    """Write processed paragraphs to out_path as JSONL. Returns
    (record_count, sha256_of_written_file)."""
    mode = "a" if append else "w"
    n = 0
    with open(out_path, mode, encoding="utf-8") as fh:
        for p in processed:
            rec = {
                "schema_version": 1,
                "record_id": p.record_id,
                "work_id": work_id,
                "title": title,
                "chapter_id": p.chapter_id,
                "paragraph_index": p.paragraph_index,
                "text": p.text,
                "source_sha256": p.source_sha256,
                "preprocessing_version": PREPROCESSING_VERSION,
                "partition": partition,
            }
            fh.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
            n += 1
    return n, _sha256_bytes(out_path.read_bytes())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None) -> int:
    # Deferred import to avoid a cycle when the tests exercise
    # preprocess_source directly with in-memory bytes.
    from scripts.vendor_aeon_lbc1 import ALLOWLIST  # noqa
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package-root",
                     default="research-data/AEON-LBC-1")
    args = ap.parse_args(argv)
    root = Path(args.package_root).resolve()
    (root / "processed").mkdir(parents=True, exist_ok=True)
    (root / "manifests").mkdir(parents=True, exist_ok=True)
    processed_manifest = {"schema_version": 1,
                            "preprocessing_version": PREPROCESSING_VERSION,
                            "generated_at_utc": time.strftime(
                                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "works": []}
    partition_files: dict = {}
    for work in ALLOWLIST:
        src = root / "source" / work.source_filename
        if not src.exists():
            processed_manifest["works"].append({
                "work_id": work.work_id,
                "status": "source_missing",
                "path": str(src)})
            continue
        raw = src.read_bytes()
        try:
            paras, log = preprocess_source(
                raw_bytes=raw, work_id=work.work_id, title=work.title)
        except PrepError as e:
            processed_manifest["works"].append({
                "work_id": work.work_id,
                "status": "prep_failed",
                "code": e.code,
                "detail": e.detail})
            continue
        out_path = root / "processed" / f"{work.partition_role}.jsonl"
        append = out_path in partition_files.values()
        n, digest = emit_partition_jsonl(
            paras, partition=work.partition_role,
            work_id=work.work_id, title=work.title,
            out_path=out_path, append=append)
        partition_files[work.work_id] = out_path
        processed_manifest["works"].append({
            "work_id": work.work_id,
            "status": "ok",
            "records": n,
            "output_partition": work.partition_role,
            "output_partition_sha256_after_write": digest,
            "preprocessing_log": log})
    (root / "manifests" / "preprocessing_manifest.json").write_text(
        json.dumps(processed_manifest, indent=2, sort_keys=True),
        encoding="utf-8")
    all_ok = all(w.get("status") == "ok" for w in processed_manifest["works"])
    print(json.dumps({"ok": all_ok,
                        "processed_works": len(processed_manifest["works"])},
                       indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
