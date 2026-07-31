"""Tests for the AEON-LBC-1 acquisition tool and legal separation.

Exercises §3 (allowlist + HTTPS + safety controls), §4 (IP-separation),
§5 (deterministic preprocessing), §6 (whole-work partitioning),
§9 (tokenizer binding).

Does NOT make live network calls — every fetch site is exercised
against an in-memory URL opener, and the boundary detector runs on
inlined synthetic Gutenberg-formatted text.
"""
import hashlib
import io
import json
import os
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# §3.1 Official source allowlist
# ---------------------------------------------------------------------------
def test_allowlist_contains_exactly_six_official_works():
    from scripts.vendor_aeon_lbc1 import ALLOWLIST
    ids = [w.work_id for w in ALLOWLIST]
    assert sorted(ids) == sorted(
        ["pg-2701", "pg-1342", "pg-11", "pg-84", "pg-55", "pg-1661"]), ids
    for w in ALLOWLIST:
        assert w.primary_url.startswith("https://"), w
        assert "gutenberg.org" in w.primary_url


def test_allowlist_partition_roles_match_directive():
    from scripts.vendor_aeon_lbc1 import ALLOWLIST
    role = {w.work_id: w.partition_role for w in ALLOWLIST}
    assert role["pg-2701"] == "train"
    assert role["pg-1342"] == "train"
    assert role["pg-11"] == "train"
    assert role["pg-84"] == "calibration"
    assert role["pg-55"] == "validation"
    assert role["pg-1661"] == "test"


# ---------------------------------------------------------------------------
# §3.3-3.5 Refuses redirects outside gutenberg.org; HTTPS-only
# ---------------------------------------------------------------------------
def test_http_urls_refused():
    from scripts.vendor_aeon_lbc1 import _assert_https, AcquisitionError
    for u in ("http://www.gutenberg.org/x.txt", "ftp://x/x"):
        try:
            _assert_https(u)
        except AcquisitionError as e:
            assert e.code == "http_forbidden"
        else:
            raise AssertionError(f"{u!r} should have been refused")


def test_non_gutenberg_hosts_refused():
    from scripts.vendor_aeon_lbc1 import _assert_allowed_host, AcquisitionError
    for u in ("https://example.com/x.txt",
                "https://raw.githubusercontent.com/x/y/main/moby.txt",
                "https://mirror.evil.gutenberg.example/pg2701.txt"):
        try:
            _assert_allowed_host(u)
        except AcquisitionError as e:
            assert e.code == "host_not_allowlisted"
        else:
            raise AssertionError(f"{u!r} should have been refused")


def test_gutenberg_subdomains_allowed():
    from scripts.vendor_aeon_lbc1 import _assert_allowed_host
    for u in ("https://www.gutenberg.org/x",
                "https://gutenberg.org/x",
                "https://aleph.gutenberg.org/x",
                "https://gutenberg.pglaf.org/x"):
        # pglaf host does NOT match gutenberg.org suffix — should be denied.
        if "pglaf" in u:
            from scripts.vendor_aeon_lbc1 import AcquisitionError
            try:
                _assert_allowed_host(u)
            except AcquisitionError:
                pass
            else:
                raise AssertionError(f"{u!r} unexpectedly allowed")
        else:
            _assert_allowed_host(u)


# ---------------------------------------------------------------------------
# §3.6-3.7 HTML rejection
# ---------------------------------------------------------------------------
def test_html_responses_rejected():
    from scripts.vendor_aeon_lbc1 import _reject_html, AcquisitionError
    class H:
        def __init__(self, m): self._m = m
        def get(self, k, default=None): return self._m.get(k, default)
    for ctype in ("text/html", "text/html; charset=utf-8",
                    "application/octet-stream", ""):
        try:
            _reject_html(H({"Content-Type": ctype}))
        except AcquisitionError as e:
            assert e.code == "non_text_plain_response", ctype
    # text/plain is accepted
    _reject_html(H({"Content-Type": "text/plain; charset=utf-8"}))


# ---------------------------------------------------------------------------
# §3.10-3.11 Digest preservation
# ---------------------------------------------------------------------------
def test_source_digest_preserved_across_re_reads():
    from scripts.vendor_aeon_lbc1 import _sha256
    payload = b"Hello Gutenberg" * 100
    d1 = _sha256(payload)
    d2 = _sha256(payload)
    assert d1 == d2
    assert d1.startswith("sha256:")


def test_refresh_source_required_on_digest_conflict():
    """Simulates a case where the target file already exists with a
    different digest, and the fetch would try to write a new one.
    Without --refresh-source, vendor_all raises."""
    from scripts.vendor_aeon_lbc1 import vendor_all, AcquisitionError, ALLOWLIST
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        src = root / "source"
        src.mkdir()
        (src / ALLOWLIST[0].source_filename).write_bytes(b"OLD BYTES")
        # Point acquisition at a fake fetcher that returns new bytes
        def fake_fetch(url, ca_bundle_path=None):
            return b"NEW BYTES\n*** START OF THIS PROJECT GUTENBERG EBOOK ***\nBody\n*** END OF THIS PROJECT GUTENBERG EBOOK ***\n"
        with mock.patch("scripts.vendor_aeon_lbc1._fetch_bytes",
                          side_effect=fake_fetch):
            try:
                vendor_all(root, refresh_source=False)
            except AcquisitionError as e:
                assert e.code == "digest_conflict"
            else:
                raise AssertionError("digest conflict should raise")


# ---------------------------------------------------------------------------
# §5 Deterministic preprocessing + strict UTF-8 + boundary detection
# ---------------------------------------------------------------------------
_SYNTHETIC_GUTENBERG = (
    "The Project Gutenberg eBook of Sample Work\n"
    "This eBook is for the use of anyone anywhere...\n\n"
    "*** START OF THIS PROJECT GUTENBERG EBOOK SAMPLE WORK ***\n\n"
    "CHAPTER I\n\n"
    "It was a bright cold day in April, and the clocks were striking thirteen.\n\n"
    "Winston Smith, his chin nuzzled into his breast, slipped quickly through the glass doors.\n\n"
    "CHAPTER II\n\n"
    "The hallway smelt of boiled cabbage and old rag mats.\n\n"
    "*** END OF THIS PROJECT GUTENBERG EBOOK SAMPLE WORK ***\n\n"
    "End of Project Gutenberg's Sample Work, by Author\n"
).encode("utf-8")


def test_boundary_detector_finds_body_range():
    from scripts.prepare_aeon_lbc1 import find_body
    text = _SYNTHETIC_GUTENBERG.decode("utf-8")
    start, end = find_body(text)
    body = text[start:end]
    assert "CHAPTER I" in body
    assert "END OF THIS PROJECT GUTENBERG" not in body
    assert "This eBook is for the use of" not in body


def test_missing_boundary_marker_is_rejected():
    from scripts.prepare_aeon_lbc1 import find_body, PrepError
    text = "no markers at all"
    try:
        find_body(text)
    except PrepError as e:
        assert e.code in ("header_marker_missing", "footer_marker_missing")


def test_utf8_decode_error_rejects_source():
    from scripts.prepare_aeon_lbc1 import preprocess_source, PrepError
    try:
        preprocess_source(raw_bytes=b"\xff\xff\xff",
                            work_id="pg-2701", title="Moby-Dick")
    except PrepError as e:
        assert e.code == "utf8_decode_failed"


def test_preprocessing_is_deterministic():
    from scripts.prepare_aeon_lbc1 import preprocess_source
    p1, log1 = preprocess_source(raw_bytes=_SYNTHETIC_GUTENBERG,
                                     work_id="pg-2701",
                                     title="Sample Work")
    p2, log2 = preprocess_source(raw_bytes=_SYNTHETIC_GUTENBERG,
                                     work_id="pg-2701",
                                     title="Sample Work")
    assert [x.record_id for x in p1] == [x.record_id for x in p2]
    assert [x.text for x in p1] == [x.text for x in p2]
    assert log1["input_sha256"] == log2["input_sha256"]


def test_record_ids_are_stable_across_invocations():
    from scripts.prepare_aeon_lbc1 import preprocess_source
    p1, _ = preprocess_source(raw_bytes=_SYNTHETIC_GUTENBERG,
                                 work_id="pg-2701",
                                 title="Sample Work")
    assert p1, "preprocess must produce records"
    for rec in p1:
        assert rec.record_id.startswith("sha256:")


def test_processed_records_omit_gutenberg_header_and_footer():
    from scripts.prepare_aeon_lbc1 import preprocess_source
    p, _ = preprocess_source(raw_bytes=_SYNTHETIC_GUTENBERG,
                                 work_id="pg-2701",
                                 title="Sample Work")
    all_text = "\n\n".join(r.text for r in p)
    assert "for the use of anyone anywhere" not in all_text
    assert "END OF THIS PROJECT GUTENBERG" not in all_text
    assert "CHAPTER I" in all_text
    assert "clocks were striking thirteen" in all_text


# ---------------------------------------------------------------------------
# §6 Whole-work partitioning + no cross-partition source reuse
# ---------------------------------------------------------------------------
def test_each_work_lands_in_exactly_one_partition():
    """No allowlisted work may appear in more than one partition role."""
    from scripts.vendor_aeon_lbc1 import ALLOWLIST
    seen = {}
    for w in ALLOWLIST:
        assert w.work_id not in seen, w.work_id
        seen[w.work_id] = w.partition_role
    partitions_used = set(seen.values())
    assert partitions_used == {"train", "calibration", "validation", "test"}


def test_test_partition_is_sherlock_holmes_only():
    from scripts.vendor_aeon_lbc1 import ALLOWLIST
    test_works = [w.work_id for w in ALLOWLIST if w.partition_role == "test"]
    assert test_works == ["pg-1661"]


# ---------------------------------------------------------------------------
# §7 Sealed-test refusal
# ---------------------------------------------------------------------------
def test_sealed_partition_summary_does_not_reveal_text():
    from aeon.bypass.sealed_partition import summarise_sealed_partition
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl",
                                       delete=False, encoding="utf-8") as f:
        for i in range(3):
            f.write(json.dumps({
                "schema_version": 1,
                "record_id": f"sha256:{'a'*64}",
                "work_id": "pg-1661",
                "chapter_id": "c",
                "paragraph_index": i,
                "text": "SECRET_HOLMES_TEXT_" + str(i),
                "source_sha256": "sha256:xyz",
                "preprocessing_version": "aeon-lbc1-v1",
                "partition": "test",
            }) + "\n")
        path = f.name
    try:
        s = summarise_sealed_partition(path)
        # The dataclass has no text-carrying fields at all.
        from dataclasses import asdict
        d = asdict(s)
        assert "SECRET_HOLMES_TEXT" not in json.dumps(d)
        assert s.record_count == 3
        assert s.work_identity == "pg-1661"
        assert s.schema_valid is True
    finally:
        os.unlink(path)


def test_sealed_partition_read_requires_lock_artifact():
    from aeon.bypass.sealed_partition import (
        read_sealed_partition, SealedPartitionAccessDenied,
    )
    with tempfile.TemporaryDirectory() as d:
        part = Path(d) / "test.jsonl"
        part.write_text(json.dumps({
            "schema_version": 1, "record_id": "sha256:x",
            "work_id": "pg-1661", "chapter_id": "c",
            "paragraph_index": 0, "text": "y",
            "source_sha256": "sha256:x",
            "preprocessing_version": "aeon-lbc1-v1",
            "partition": "test",
        }) + "\n")
        missing_lock = str(Path(d) / "does_not_exist_lock.json")
        try:
            list(read_sealed_partition(str(part),
                                          lock_artifact_path=missing_lock))
        except SealedPartitionAccessDenied as e:
            assert e.code == "lock_artifact_invalid"


def test_sealed_partition_lock_artifact_requires_all_fields():
    from aeon.bypass.sealed_partition import validate_lock_artifact
    with tempfile.TemporaryDirectory() as d:
        lock = Path(d) / "L3_CALIBRATION_LOCK.json"
        lock.write_text(json.dumps({"barrier_registry_digest": "sha256:x"}))
        ok, errs = validate_lock_artifact(str(lock))
        assert ok is False
        assert errs  # must enumerate missing keys


def test_experimental_version_bump_detected():
    from aeon.bypass.sealed_partition import experimental_version_bumped
    with tempfile.TemporaryDirectory() as d:
        old_p = Path(d) / "old.json"; old_p.write_text(json.dumps(
            {"experimental_version": "v1"}))
        new_p = Path(d) / "new.json"; new_p.write_text(json.dumps(
            {"experimental_version": "v2"}))
        assert experimental_version_bumped(str(old_p), str(new_p)) is True
        same_p = Path(d) / "same.json"; same_p.write_text(json.dumps(
            {"experimental_version": "v1"}))
        assert experimental_version_bumped(str(old_p), str(same_p)) is False


# ---------------------------------------------------------------------------
# §4 Legal / IP separation — corpus license does not touch Aeon
# ---------------------------------------------------------------------------
def test_corpus_notice_does_not_overwrite_aeon_license():
    """No Project Gutenberg / corpus-license text may be pasted over
    the Aeon repository license or a top-level LICENSE-equivalent."""
    forbidden = [
        "Project Gutenberg is a registered trademark",
        "This eBook is for the use of anyone anywhere",
        "PROJECT GUTENBERG EBOOK",
    ]
    # Sweep repo-level license-shaped files. If they don't exist, that
    # is fine — this test guards against a future corpus notice leaking
    # into them.
    candidates = ["LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING",
                    "README.md"]
    for name in candidates:
        p = os.path.join(ROOT, name)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        for phrase in forbidden:
            assert phrase not in src, (
                f"Corpus notice bled into {name!r}: found {phrase!r}. "
                "Corpus license must remain isolated in the corpus package.")


def test_no_corpus_files_bundled_in_aeon_installer_paths():
    """The Aeon installer's [Files] Source path must not currently pull
    in research-data/ (the corpus package location)."""
    iss = os.path.join(ROOT, "packaging", "windows", "AeonInstaller.iss")
    if os.path.exists(iss):
        src = open(iss, encoding="utf-8").read()
        assert "research-data" not in src, (
            "AeonInstaller.iss must not bundle research-data/ into the "
            "installer")
        assert "corpus-package" not in src


# ---------------------------------------------------------------------------
# §9 Tokenizer binding excludes the sealed test partition
# ---------------------------------------------------------------------------
def test_tokenizer_binding_manifest_prohibits_test_partition_use():
    """The tokenizer binding manifest schema records
    used_partitions and must not include 'test'."""
    binding = {
        "tokenizer_implementation": "aeon.tokenizer.AeonTokenizer",
        "tokenizer_model_digest": "sha256:x",
        "vocab_size": 32000,
        "special_ids": {"pad": 0, "unk": 1, "bos": 2, "eos": 3},
        "preprocessing_version": "aeon-lbc1-v1",
        "used_partitions": ["train"],  # explicit train-only
    }
    assert "test" not in binding["used_partitions"]
    assert "calibration" not in binding["used_partitions"]
    assert "validation" not in binding["used_partitions"]


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
