"""
aeon/corpus_manifest.py — schema + validator for the corpus source manifest.

Schema: docs/corpus_manifest_schema.json. Refusal rules (F2.4/F2.5):
  - Any included source with a missing required field ⇒ ProvenanceError.
  - trust_level=quarantined AND partition in {train, validation} ⇒ ProvenanceError.
  - inclusion_status=excluded AND no rejection_reason ⇒ ProvenanceError.
  - The bulk `content_sha256` field must be recomputed and matched by the
    caller when the physical file is available (a helper is provided).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

from aeon.provenance import ProvenanceError, hash_file


REQUIRED_FIELDS = [
    "source_id", "origin", "acquired_at", "license_status",
    "content_sha256", "preprocessing_version", "filtering_version",
    "deduplication_version", "partition_assignment", "inclusion_status",
    "rejection_reason_if_rejected", "trust_level",
]
TRUST_LEVELS = {"trusted", "untrusted", "quarantined"}
INCLUSION_STATUSES = {"included", "excluded", "quarantined_pending"}
PARTITIONS = {"train", "validation", "held_out", "eval_only"}


def validate_source(src: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    for f in REQUIRED_FIELDS:
        if f not in src:
            errs.append(f"source {src.get('source_id', '?')}: missing {f}")
    tl = src.get("trust_level")
    if tl not in TRUST_LEVELS:
        errs.append(f"source {src.get('source_id', '?')}: trust_level={tl!r} not in {sorted(TRUST_LEVELS)}")
    inc = src.get("inclusion_status")
    if inc not in INCLUSION_STATUSES:
        errs.append(f"source {src.get('source_id', '?')}: inclusion_status={inc!r}")
    part = src.get("partition_assignment")
    if part not in PARTITIONS:
        errs.append(f"source {src.get('source_id', '?')}: partition_assignment={part!r}")
    # Quarantined sources may not enter train/validation.
    if tl == "quarantined" and part in {"train", "validation"}:
        errs.append(f"source {src.get('source_id', '?')}: quarantined sources may not enter {part}")
    # Excluded sources must state a reason.
    if inc == "excluded" and not src.get("rejection_reason_if_rejected"):
        errs.append(f"source {src.get('source_id', '?')}: excluded without rejection_reason")
    return errs


def validate_manifest(manifest: Dict[str, Any]) -> List[str]:
    """Return list of errors; empty means valid. Enforces F2.4/F2.5."""
    errs: List[str] = []
    if "sources" not in manifest or not isinstance(manifest["sources"], list):
        errs.append("manifest missing 'sources' list")
        return errs
    seen_ids: set = set()
    for src in manifest["sources"]:
        errs += validate_source(src)
        if src.get("source_id") in seen_ids:
            errs.append(f"duplicate source_id: {src.get('source_id')}")
        seen_ids.add(src.get("source_id"))
    return errs


def refuse_if_invalid(manifest: Dict[str, Any]) -> None:
    """Fail-closed: raise ProvenanceError with the first blocking issue."""
    errs = validate_manifest(manifest)
    if errs:
        raise ProvenanceError("; ".join(errs[:5]))


def verify_source_content(src: Dict[str, Any], filesystem_path: str) -> None:
    """Recompute content_sha256 against the file and refuse on mismatch."""
    expected = src.get("content_sha256")
    if not expected:
        raise ProvenanceError(f"source {src.get('source_id')}: no content_sha256")
    actual = hash_file(filesystem_path)
    if expected != actual:
        raise ProvenanceError(
            f"source {src.get('source_id')}: sha256 mismatch expected={expected} actual={actual}")


def synthetic_manifest_for_smoke() -> Dict[str, Any]:
    """A single-source manifest used by the synthetic-token training smoke
    when no real corpus is configured. Records EXACTLY that it is synthetic."""
    return {
        "sources": [{
            "source_id": "synthetic_random_tokens",
            "origin": "in-memory Torch random tokens (aeon/scripts train.py fallback)",
            "acquired_at": "0000-00-00",
            "license_status": "not_applicable_synthetic",
            "content_sha256": "0" * 64,
            "preprocessing_version": "n/a",
            "filtering_version": "n/a",
            "deduplication_version": "n/a",
            "partition_assignment": "train",
            "inclusion_status": "included",
            "rejection_reason_if_rejected": None,
            "trust_level": "trusted",
        }]
    }
