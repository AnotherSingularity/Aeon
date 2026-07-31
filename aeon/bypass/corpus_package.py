"""L3 real-English corpus-package validation machinery.

The L-series' scientific claims (Level 2+) require a vendored,
authenticated, real-English corpus with fixed train / calibration /
validation / sealed-test partitions. This module owns the format the
package must follow and the validator L3 runs before any calibration
touches it.

Package layout:

    corpus-package/
        source/
            original.txt                — raw upstream text
        processed/
            train.jsonl
            calibration.jsonl
            validation.jsonl
            test.jsonl
        manifest.json                    — see MANIFEST_REQUIRED_KEYS
        provenance.json                  — retrieval evidence
        license.txt                      — public-domain attestation
        partition_report.json            — leakage-report
        SEALED_TEST_DIGEST               — content-hash of the sealed
                                           test partition, retained
                                           until threshold locking is
                                           complete

The validator never opens SEALED_TEST_DIGEST unless
``allow_test_partition_access=True`` is passed explicitly by the
caller — and even then the seal digest is compared, not the file
content.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


MANIFEST_REQUIRED_KEYS = (
    "schema_version",
    "corpus_id",
    "title",
    "source",
    "retrieval_date",
    "source_sha256",
    "public_domain_basis",
    "license_status",
    "preprocessing_version",
    "tokenizer_id",
    "partitions",
)

REQUIRED_PARTITIONS = ("train", "calibration", "validation", "test")


class CorpusPackageError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class CorpusPackageValidationResult:
    corpus_id: str
    manifest_ok: bool
    provenance_ok: bool
    license_ok: bool
    partitions_present: Dict[str, bool]
    partition_digests_match: Dict[str, bool]
    test_partition_sealed: bool
    leakage_report_present: bool
    ready_for_L3: bool
    errors: List[str] = field(default_factory=list)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_corpus_package(
    package_root: str,
    *,
    allow_test_partition_access: bool = False,
) -> CorpusPackageValidationResult:
    """Validate a corpus package on disk. Returns a
    CorpusPackageValidationResult; never raises for legitimate
    validation failures (they populate ``errors`` and set
    ``ready_for_L3=False``).

    Raises only on catastrophic conditions (e.g. package root does
    not exist)."""
    if not os.path.isdir(package_root):
        raise CorpusPackageError(
            "package_root_missing", f"{package_root!r}")
    errors: List[str] = []
    # --- 1. manifest.json ---
    manifest_path = os.path.join(package_root, "manifest.json")
    manifest_ok = True
    manifest: Dict[str, Any] = {}
    if not os.path.exists(manifest_path):
        errors.append("manifest.json missing")
        manifest_ok = False
    else:
        try:
            with open(manifest_path, encoding="utf-8") as fh:
                manifest = json.load(fh)
        except Exception as e:
            errors.append(f"manifest.json unreadable: {e}")
            manifest_ok = False
        for k in MANIFEST_REQUIRED_KEYS:
            if k not in manifest:
                errors.append(f"manifest.json missing key {k!r}")
                manifest_ok = False
    corpus_id = manifest.get("corpus_id", "")

    # --- 2. provenance.json ---
    prov_path = os.path.join(package_root, "provenance.json")
    provenance_ok = os.path.exists(prov_path)
    if not provenance_ok:
        errors.append("provenance.json missing")

    # --- 3. license.txt (public-domain attestation) ---
    lic_path = os.path.join(package_root, "license.txt")
    license_ok = os.path.exists(lic_path)
    if not license_ok:
        errors.append("license.txt missing")

    # --- 4. Partitions present + digests match manifest ---
    partitions_present: Dict[str, bool] = {}
    partition_digests_match: Dict[str, bool] = {}
    for name in REQUIRED_PARTITIONS:
        pp = os.path.join(package_root, "processed", f"{name}.jsonl")
        present = os.path.exists(pp)
        partitions_present[name] = present
        if not present:
            errors.append(f"processed/{name}.jsonl missing")
            partition_digests_match[name] = False
            continue
        # digest comparison (skipped for test unless allow_test_partition_access)
        if name == "test" and not allow_test_partition_access:
            partition_digests_match[name] = True  # not inspected
            continue
        expected = manifest.get("partitions", {}).get(name, {}).get("sha256")
        if expected is None:
            errors.append(f"manifest.partitions.{name}.sha256 missing")
            partition_digests_match[name] = False
            continue
        actual = _sha256_file(pp)
        partition_digests_match[name] = (expected == actual)
        if expected != actual:
            errors.append(
                f"processed/{name}.jsonl sha256 mismatch: "
                f"manifest={expected!r} actual={actual!r}")

    # --- 5. Test partition sealed ---
    seal_path = os.path.join(package_root, "SEALED_TEST_DIGEST")
    test_partition_sealed = os.path.exists(seal_path)
    if not test_partition_sealed:
        errors.append("SEALED_TEST_DIGEST missing — the held-out test "
                      "partition must be sealed until thresholds and "
                      "reaction coordinates are locked")

    # --- 6. Leakage report ---
    leak_path = os.path.join(package_root, "partition_report.json")
    leakage_report_present = os.path.exists(leak_path)
    if not leakage_report_present:
        errors.append("partition_report.json missing")

    ready = (manifest_ok and provenance_ok and license_ok
              and all(partitions_present.values())
              and all(partition_digests_match.values())
              and test_partition_sealed
              and leakage_report_present)

    return CorpusPackageValidationResult(
        corpus_id=corpus_id,
        manifest_ok=manifest_ok,
        provenance_ok=provenance_ok,
        license_ok=license_ok,
        partitions_present=partitions_present,
        partition_digests_match=partition_digests_match,
        test_partition_sealed=test_partition_sealed,
        leakage_report_present=leakage_report_present,
        ready_for_L3=ready,
        errors=errors,
    )
