"""L3 sealed-test-partition control.

Before the L3 calibration lock is committed, code may inspect ONLY the
sealed test partition's:

    * record count
    * byte count
    * SHA-256
    * work identity
    * schema validity

It may NOT return record text or token IDs.

After ``docs/latent_bypass/L3_CALIBRATION_LOCK.json`` is committed and
valid, held-out access is enabled — but every subsequent change to
thresholds, reaction coordinate, checkpoint, tokenizer, barrier
definitions, or analysis plan invalidates the old lock and requires a
NEW experimental version identifier. Previously opened test results
are not fresh confirmatory evidence for the new version.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional


LOCK_ARTIFACT_REQUIRED_KEYS = (
    "barrier_registry_digest",
    "barrier_thresholds",
    "reaction_coordinate_specification",
    "calibration_corpus_digest",
    "model_checkpoint_identity",
    "tokenizer_identity",
    "statistical_plan",
    "intervention_plan",
    "evidence_thresholds",
    "exact_commit",
    "utc_creation_time",
    "experimental_version",
)


class SealedPartitionAccessDenied(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class SealedPartitionSummary:
    partition_path: str
    record_count: int
    byte_count: int
    sha256: str
    work_identity: Optional[str]
    schema_valid: bool
    schema_errors: List[str] = field(default_factory=list)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def summarise_sealed_partition(partition_path: str) -> SealedPartitionSummary:
    """Return schema-level facts about the sealed partition without
    revealing text. Callers holding the returned summary cannot
    reconstruct records."""
    n = 0
    schema_errors: List[str] = []
    work_id: Optional[str] = None
    with open(partition_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                schema_errors.append(f"line {line_no}: json_decode: {e}")
                continue
            if not isinstance(rec, dict):
                schema_errors.append(f"line {line_no}: not_object")
                continue
            for k in ("schema_version", "record_id", "work_id",
                        "chapter_id", "paragraph_index", "partition",
                        "preprocessing_version"):
                if k not in rec:
                    schema_errors.append(f"line {line_no}: missing {k!r}")
            wi = rec.get("work_id")
            if work_id is None:
                work_id = wi
            elif work_id != wi:
                schema_errors.append(
                    f"line {line_no}: mixed work_id in sealed partition "
                    f"({work_id!r} vs {wi!r})")
            n += 1
    st = os.stat(partition_path)
    return SealedPartitionSummary(
        partition_path=partition_path,
        record_count=n,
        byte_count=st.st_size,
        sha256=_sha256_file(partition_path),
        work_identity=work_id,
        schema_valid=not schema_errors,
        schema_errors=schema_errors,
    )


def read_sealed_partition(partition_path: str, *, lock_artifact_path: str):
    """Return an iterator over the sealed partition's records.

    REFUSES unless a valid L3 calibration-lock artifact is present.
    Every caller reading the sealed partition must go through this
    entry point — never open the file directly."""
    lock_ok, errors = validate_lock_artifact(lock_artifact_path)
    if not lock_ok:
        raise SealedPartitionAccessDenied(
            "lock_artifact_invalid",
            "; ".join(errors[:5]) + ("; …" if len(errors) > 5 else ""))
    with open(partition_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            yield json.loads(line)


def validate_lock_artifact(path: str) -> "tuple[bool, list[str]]":
    """Return (ok, errors)."""
    if not os.path.exists(path):
        return False, [f"missing: {path}"]
    try:
        with open(path, encoding="utf-8") as fh:
            payload = json.load(fh)
    except Exception as e:
        return False, [f"unreadable: {e}"]
    errors: List[str] = []
    for k in LOCK_ARTIFACT_REQUIRED_KEYS:
        if k not in payload:
            errors.append(f"missing key {k!r}")
    return (not errors), errors


def experimental_version_bumped(
    old_lock_path: str, new_lock_path: str,
) -> bool:
    """After ANY change to thresholds / reaction coordinate /
    checkpoint / tokenizer / barrier definitions / analysis plan, the
    new lock artifact must carry a distinct ``experimental_version``
    string. Returns True if the bump is present, False if the caller
    tried to reuse the old identifier."""
    if not (os.path.exists(old_lock_path) and os.path.exists(new_lock_path)):
        return False
    with open(old_lock_path, encoding="utf-8") as fh:
        old = json.load(fh)
    with open(new_lock_path, encoding="utf-8") as fh:
        new = json.load(fh)
    return old.get("experimental_version") != new.get("experimental_version")
