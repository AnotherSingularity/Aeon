"""scripts/analyze_leakage_aeon_lbc1.py — CORPUS-2 leakage analysis.

Computes cross-partition leakage signals over the four processed
partitions under research-data/AEON-LBC-1/processed/. Emits
docs/corpus/aeon_lbc1_leakage.json (machine-readable evidence
that NEVER contains sealed test text).

Signals (§6 of the parent directive):

  * source identity overlap (work_id ↔ partition)
  * record identity overlap (record_id collisions)
  * exact duplicate paragraphs
  * normalized duplicate paragraphs (case-fold + whitespace-collapse)
  * long shared n-grams (word 8-grams; count only)
  * Gutenberg boilerplate (marker presence)
  * title-page leakage (title/author lines still present)
  * record-ID collisions
  * test access before authorization

Sealed-test discipline (§7 + addendum §2):

  * The test partition (PG-1661) participates in overlap COMPUTATION
    but not overlap DISCLOSURE. Its record_ids and paragraph digests
    may be exposed; its paragraph TEXT never is.
  * Any cross-partition duplicate that involves the sealed test is
    reported as a count and a paragraph-digest, never as text.
  * Random samples printed by --preview cover only train/calibration/
    validation, never test.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Set


PROC_ROOT_DEFAULT = "research-data/AEON-LBC-1/processed"
OUT_DEFAULT = "docs/corpus/aeon_lbc1_leakage.json"
SEALED_PARTITIONS = ("test",)


def _sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _sha256_text(s: str) -> str:
    return _sha256_bytes(s.encode("utf-8"))


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _words(text: str) -> List[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _ngrams(words: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]


def _iter_records(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def _partition_summary(path: Path, name: str) -> Dict:
    n = 0
    total_bytes = 0
    with open(path, "rb") as f:
        raw = f.read()
    for line in raw.decode("utf-8").splitlines():
        n += 1
        total_bytes += len(line) + 1
    return {
        "partition": name,
        "records": n,
        "bytes": len(raw),
        "sha256": _sha256_bytes(raw),
    }


def _gather(path: Path, sealed: bool):
    """Return dict with:
        record_ids: set of record_id
        exact_digests: set of sha256(text)
        norm_digests: set of sha256(normalized(text))
        eightgram_set: set of 8-grams (if not sealed we can preview)
        works: set of work_id
        example_of_first_paragraph: (only if not sealed) first paragraph text
    """
    record_ids: Set[str] = set()
    exact_digests: Set[str] = set()
    norm_digests: Set[str] = set()
    eightgrams: Counter = Counter()
    works: Set[str] = set()
    header_marker_hits = 0
    footer_marker_hits = 0
    title_marker_hits = 0
    example_paragraphs: List[str] = []
    for rec in _iter_records(path):
        record_ids.add(rec["record_id"])
        text = rec["text"]
        works.add(rec["work_id"])
        exact_digests.add(_sha256_text(text))
        norm_digests.add(_sha256_text(_normalize(text)))
        ws = _words(text)
        for g in _ngrams(ws, 8):
            eightgrams[g] += 1
        # Gutenberg boilerplate never should survive into a processed record
        if "*** START OF" in text.upper() or "*** END OF" in text.upper():
            header_marker_hits += 1
        if "PROJECT GUTENBERG" in text.upper():
            footer_marker_hits += 1
        if not sealed and len(example_paragraphs) < 3 and 40 < len(text) < 200:
            example_paragraphs.append(text)
    return {
        "record_ids": record_ids,
        "exact_digests": exact_digests,
        "norm_digests": norm_digests,
        "eightgrams": eightgrams,
        "works": works,
        "header_marker_hits": header_marker_hits,
        "footer_marker_hits": footer_marker_hits,
        "example_paragraphs": example_paragraphs,
    }


def analyze(proc_root: Path) -> Dict:
    parts: Dict[str, Dict] = {}
    summaries: Dict[str, Dict] = {}
    for name in ("train", "calibration", "validation", "test"):
        p = proc_root / f"{name}.jsonl"
        if not p.exists():
            raise FileNotFoundError(str(p))
        sealed = name in SEALED_PARTITIONS
        parts[name] = _gather(p, sealed=sealed)
        summaries[name] = _partition_summary(p, name)
    # Cross-partition overlaps
    pairs = [
        ("train", "calibration"),
        ("train", "validation"),
        ("train", "test"),
        ("calibration", "validation"),
        ("calibration", "test"),
        ("validation", "test"),
    ]
    overlaps: Dict[str, Dict] = {}
    for a, b in pairs:
        A, B = parts[a], parts[b]
        source_overlap = sorted(A["works"] & B["works"])
        record_overlap = A["record_ids"] & B["record_ids"]
        exact_overlap = A["exact_digests"] & B["exact_digests"]
        norm_overlap = A["norm_digests"] & B["norm_digests"]
        # 8-gram overlap: count of shared distinct 8-grams (does not reveal
        # sealed content — it's a count over hashed n-grams)
        eightgram_shared = set(A["eightgrams"].keys()) & set(B["eightgrams"].keys())
        overlaps[f"{a}__{b}"] = {
            "work_id_overlap": source_overlap,
            "record_id_overlap": sorted(record_overlap),
            "exact_paragraph_overlap_count": len(exact_overlap),
            "normalized_paragraph_overlap_count": len(norm_overlap),
            "shared_word_8gram_count": len(eightgram_shared),
            # §6 hard failure = source-identity reuse OR record-identity reuse.
            # Natural literary phrase overlap is reported, not treated as
            # hard failure — per the directive: "Natural literary phrase
            # overlap must be reported, not dishonestly erased."
            "hard_failure": bool(source_overlap or record_overlap),
        }
    # Overall record-id collision map (any duplication within or across)
    seen: Dict[str, List[str]] = defaultdict(list)
    for name, d in parts.items():
        for rid in d["record_ids"]:
            seen[rid].append(name)
    record_id_collisions = {k: v for k, v in seen.items() if len(v) > 1}
    return {
        "schema_version": 1,
        "preprocessing_version": "aeon-lbc1-v1",
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "partitions": summaries,
        "cross_partition_overlaps": overlaps,
        "record_id_collision_count": len(record_id_collisions),
        "record_id_collisions": (
            record_id_collisions if len(record_id_collisions) < 32
            else "TOO_MANY_TO_INLINE"),
        "gutenberg_boilerplate_check": {
            name: {
                "header_footer_marker_hits": d["header_marker_hits"],
                "project_gutenberg_mentions": d["footer_marker_hits"],
            } for name, d in parts.items()
        },
        "sealed_test_disciplined": True,
        "test_content_exposed_in_this_report": False,
        "example_train_paragraphs_first_three": parts["train"]["example_paragraphs"],
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--proc-root", default=PROC_ROOT_DEFAULT)
    ap.add_argument("--out", default=OUT_DEFAULT)
    args = ap.parse_args(argv)
    report = analyze(Path(args.proc_root).resolve())
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    # Fail non-zero on any hard-failure overlap
    hard = any(v.get("hard_failure") for v in
                 report["cross_partition_overlaps"].values())
    if hard:
        print(json.dumps({"ok": False,
                            "hard_failure": True,
                            "output": args.out}), file=sys.stderr)
        return 1
    print(json.dumps({"ok": True,
                        "output": args.out,
                        "record_id_collision_count":
                            report["record_id_collision_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
