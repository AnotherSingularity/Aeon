"""Tests for the AEON-LBC-1 offline intake mode (§1 / §4 / §5).

Every test operates entirely offline — no live network, no fake HTTP
server. Sources are synthesised in a tmpdir per test.
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------
_TITLES = {
    "pg-0011.txt": "Alice's Adventures in Wonderland",
    "pg-0055.txt": "The Wonderful Wizard of Oz",
    "pg-0084.txt": "Frankenstein; Or, The Modern Prometheus",
    "pg-1342.txt": "Pride and Prejudice",
    "pg-1661.txt": "The Adventures of Sherlock Holmes",
    "pg-2701.txt": "Moby-Dick; Or, The Whale",
}
_EBOOK_IDS = {
    "pg-0011.txt": 11,
    "pg-0055.txt": 55,
    "pg-0084.txt": 84,
    "pg-1342.txt": 1342,
    "pg-1661.txt": 1661,
    "pg-2701.txt": 2701,
}
# Content-identity evidence phrases the vendor script's
# _WORK_TITLE_EVIDENCE table looks for.
_EVIDENCE_TAG = {
    "pg-0011.txt": "Alice's Adventures in Wonderland",
    "pg-0055.txt": "The Wonderful Wizard of Oz",
    "pg-0084.txt": "Frankenstein",
    "pg-1342.txt": "Pride and Prejudice",
    "pg-1661.txt": "The Adventures of Sherlock Holmes",
    "pg-2701.txt": "Moby-Dick",
}


def _gutenberg_source_body(filename: str) -> bytes:
    title = _TITLES[filename]
    ebook_id = _EBOOK_IDS[filename]
    tag = _EVIDENCE_TAG[filename]
    body = f"""The Project Gutenberg eBook of {title}

Title: {title}
Author: Test Author
Release date: EBook #{ebook_id}

*** START OF THIS PROJECT GUTENBERG EBOOK {tag.upper()} ***

CHAPTER I

{tag} is a well known work. The following body text is a synthetic
stand-in used only to exercise the offline intake pipeline. No
scientific claim is derived from it.

CHAPTER II

More body text. Enough paragraphs to survive the header/footer
stripping stage without producing an empty processed partition.

*** END OF THIS PROJECT GUTENBERG EBOOK {tag.upper()} ***

End of Project Gutenberg's {title}, by Test Author
"""
    return body.encode("utf-8")


def _provenance(filename: str,
                  *,
                  override_ebook_id: int = None,
                  override_title: str = None,
                  override_format: str = None,
                  override_method: str = None,
                  override_public_domain: str = None,
                  drop_key: str = None) -> dict:
    prov = {
        "schema_version": 1,
        "ebook_id": override_ebook_id or _EBOOK_IDS[filename],
        "title": override_title or _TITLES[filename],
        "source_provider": "Project Gutenberg",
        "source_page": f"https://www.gutenberg.org/ebooks/{_EBOOK_IDS[filename]}",
        "retrieval_date": "2026-08-01",
        "retrieval_method": override_method or "manual_browser_download",
        "format": override_format or "plain_text_utf8",
        "public_domain_basis": override_public_domain or "public_domain_in_usa",
        "provided_by": "repository_owner",
    }
    if drop_key:
        prov.pop(drop_key, None)
    return prov


def _build_intake(tmp: Path, *,
                    skip_files: set = None,
                    extra_files: dict = None,
                    override_bytes: dict = None,
                    override_provenance: dict = None,
                    skip_provenance: set = None) -> Path:
    """Create an intake directory. Optional hooks let tests exercise
    negative cases."""
    skip_files = skip_files or set()
    extra_files = extra_files or {}
    override_bytes = override_bytes or {}
    override_provenance = override_provenance or {}
    skip_provenance = skip_provenance or set()
    intake = tmp / "incoming"
    (intake / "sources").mkdir(parents=True)
    (intake / "provenance").mkdir(parents=True)
    for filename in _TITLES.keys():
        if filename in skip_files:
            continue
        data = override_bytes.get(filename, _gutenberg_source_body(filename))
        (intake / "sources" / filename).write_bytes(data)
        if filename in skip_provenance:
            continue
        prov = override_provenance.get(filename, _provenance(filename))
        prov_name = filename.rsplit(".", 1)[0] + ".json"
        (intake / "provenance" / prov_name).write_text(
            json.dumps(prov, sort_keys=True), encoding="utf-8")
    for extra_name, extra_data in extra_files.items():
        (intake / "sources" / extra_name).write_bytes(extra_data)
    return intake


# ---------------------------------------------------------------------------
# §1 — Network access is not attempted under --source-dir
# ---------------------------------------------------------------------------
def test_offline_mode_does_not_attempt_network_fetch():
    """The offline path must not go through _fetch_bytes at all. Patch
    it to raise; a passing test proves the network code was never
    reached."""
    import unittest.mock as mock
    from scripts import vendor_aeon_lbc1 as V
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        intake = _build_intake(tmp)
        package_root = tmp / "AEON-LBC-1"
        def _explode(*a, **kw):
            raise AssertionError("offline mode must not call the network")
        with mock.patch.object(V, "_fetch_bytes", side_effect=_explode):
            summary = V.import_offline_sources(package_root, intake)
        assert summary["acquisition_method"] == "manual_official_download"
        assert summary["executing_environment_did_not_download"] is True


# ---------------------------------------------------------------------------
# §1.2 — Exactly six sources; extras rejected; missing rejected
# ---------------------------------------------------------------------------
def test_exactly_six_sources_required():
    from scripts import vendor_aeon_lbc1 as V
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        intake = _build_intake(tmp)
        package_root = tmp / "AEON-LBC-1"
        summary = V.import_offline_sources(package_root, intake)
        assert len(summary["works"]) == 6


def test_extra_sources_rejected():
    from scripts import vendor_aeon_lbc1 as V
    from scripts.vendor_aeon_lbc1 import AcquisitionError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        intake = _build_intake(
            tmp,
            extra_files={"pg-9999.txt": b"This is not an allowlisted work."})
        try:
            V.import_offline_sources(tmp / "AEON-LBC-1", intake)
        except AcquisitionError as e:
            assert e.code == "intake_unexpected_sources"
            assert "pg-9999.txt" in e.detail
        else:
            raise AssertionError("extras must be refused")


def test_missing_sources_rejected():
    from scripts import vendor_aeon_lbc1 as V
    from scripts.vendor_aeon_lbc1 import AcquisitionError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        intake = _build_intake(tmp, skip_files={"pg-2701.txt"})
        try:
            V.import_offline_sources(tmp / "AEON-LBC-1", intake)
        except AcquisitionError as e:
            assert e.code == "intake_missing_sources"
            assert "pg-2701.txt" in e.detail
        else:
            raise AssertionError("missing sources must be refused")


# ---------------------------------------------------------------------------
# §1.6-7 — Content identity: gutenberg markers + title evidence
# ---------------------------------------------------------------------------
def test_missing_gutenberg_marker_rejected():
    from scripts import vendor_aeon_lbc1 as V
    from scripts.vendor_aeon_lbc1 import AcquisitionError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Same body but with the marker line stripped.
        broken = b"Alice is walking down the road.\n"
        intake = _build_intake(tmp, override_bytes={"pg-0011.txt": broken})
        try:
            V.import_offline_sources(tmp / "AEON-LBC-1", intake)
        except AcquisitionError as e:
            assert e.code == "no_gutenberg_header"
        else:
            raise AssertionError("missing gutenberg marker must be refused")


def test_wrong_work_content_rejected():
    """A file whose content-title differs from the filename mapping
    (e.g. pg-2701.txt containing Alice content) must be refused."""
    from scripts import vendor_aeon_lbc1 as V
    from scripts.vendor_aeon_lbc1 import AcquisitionError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wrong = _gutenberg_source_body("pg-0011.txt")  # Alice content
        intake = _build_intake(tmp, override_bytes={"pg-2701.txt": wrong})
        try:
            V.import_offline_sources(tmp / "AEON-LBC-1", intake)
        except AcquisitionError as e:
            assert e.code == "title_evidence_missing"
        else:
            raise AssertionError("filename/content mismatch must be refused")


# ---------------------------------------------------------------------------
# §1.8 — HTML / PDF / binary masquerade rejection
# ---------------------------------------------------------------------------
def test_html_masquerading_as_text_rejected():
    from scripts import vendor_aeon_lbc1 as V
    from scripts.vendor_aeon_lbc1 import AcquisitionError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        html = b"<!DOCTYPE html>\n<html><head><title>Moby-Dick</title></head></html>"
        intake = _build_intake(tmp, override_bytes={"pg-2701.txt": html})
        try:
            V.import_offline_sources(tmp / "AEON-LBC-1", intake)
        except AcquisitionError as e:
            assert e.code == "intake_non_plain_text"
        else:
            raise AssertionError("HTML masquerade must be refused")


def test_pdf_masquerading_as_text_rejected():
    from scripts import vendor_aeon_lbc1 as V
    from scripts.vendor_aeon_lbc1 import AcquisitionError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pdf = b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n"
        intake = _build_intake(tmp, override_bytes={"pg-84.txt": pdf,
                                                          "pg-0084.txt": pdf})
        try:
            V.import_offline_sources(tmp / "AEON-LBC-1", intake)
        except AcquisitionError as e:
            assert e.code == "intake_non_plain_text"
        else:
            raise AssertionError("PDF masquerade must be refused")


def test_binary_data_rejected():
    from scripts import vendor_aeon_lbc1 as V
    from scripts.vendor_aeon_lbc1 import AcquisitionError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Includes a real NUL byte in the first BINARY_SNIFF_BYTES.
        bin_ = b"\x00\x01\x02\x03random bytes not text"
        intake = _build_intake(tmp, override_bytes={"pg-1661.txt": bin_})
        try:
            V.import_offline_sources(tmp / "AEON-LBC-1", intake)
        except AcquisitionError as e:
            assert e.code == "intake_binary_data"
        else:
            raise AssertionError("binary payload must be refused")


# ---------------------------------------------------------------------------
# §1.5 — Strict UTF-8
# ---------------------------------------------------------------------------
def test_invalid_utf8_rejected():
    from scripts import vendor_aeon_lbc1 as V
    from scripts.vendor_aeon_lbc1 import AcquisitionError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # Prefix with plausible-looking header, then non-UTF-8 bytes.
        bad = (b"The Project Gutenberg eBook of Test\n"
                b"*** START OF THIS PROJECT GUTENBERG EBOOK ***\n"
                b"\xff\xfe not utf-8")
        intake = _build_intake(tmp, override_bytes={"pg-1342.txt": bad})
        try:
            V.import_offline_sources(tmp / "AEON-LBC-1", intake)
        except AcquisitionError as e:
            assert e.code in ("intake_utf8_decode_failed",
                                "intake_binary_data")
        else:
            raise AssertionError("invalid UTF-8 must be refused")


# ---------------------------------------------------------------------------
# §1.10 — SHA-256 recorded from supplied bytes; modifying source
#          changes package identity
# ---------------------------------------------------------------------------
def test_source_digest_binds_downstream_artifacts():
    from scripts import vendor_aeon_lbc1 as V
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        intake = _build_intake(tmp)
        pkg = tmp / "AEON-LBC-1"
        s1 = V.import_offline_sources(pkg, intake)
        digest_map_1 = json.loads(
            (pkg / "ORIGINAL_SOURCE_DIGESTS").read_text())
        # Now modify one incoming source, rebuild.
        pkg2_root = tmp / "AEON-LBC-1-alt"
        intake2 = _build_intake(
            tmp / "alt",
            override_bytes={"pg-0011.txt":
                             _gutenberg_source_body("pg-0011.txt") + b"\nAdded paragraph.\n"})
        s2 = V.import_offline_sources(pkg2_root, intake2)
        digest_map_2 = json.loads(
            (pkg2_root / "ORIGINAL_SOURCE_DIGESTS").read_text())
        assert (digest_map_1["pg-11"]["sha256"]
                != digest_map_2["pg-11"]["sha256"])


# ---------------------------------------------------------------------------
# §1.14 — Missing provenance rejected
# ---------------------------------------------------------------------------
def test_missing_provenance_sidecar_rejected():
    from scripts import vendor_aeon_lbc1 as V
    from scripts.vendor_aeon_lbc1 import AcquisitionError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        intake = _build_intake(tmp, skip_provenance={"pg-1342.txt"})
        try:
            V.import_offline_sources(tmp / "AEON-LBC-1", intake)
        except AcquisitionError as e:
            assert e.code == "intake_provenance_missing"
        else:
            raise AssertionError("missing provenance must be refused")


def test_invalid_retrieval_method_rejected():
    from scripts import vendor_aeon_lbc1 as V
    from scripts.vendor_aeon_lbc1 import AcquisitionError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bad_prov = _provenance("pg-1342.txt", override_method="scraper_bot")
        intake = _build_intake(tmp,
                                override_provenance={"pg-1342.txt": bad_prov})
        try:
            V.import_offline_sources(tmp / "AEON-LBC-1", intake)
        except AcquisitionError as e:
            assert e.code == "provenance_invalid_retrieval_method"
        else:
            raise AssertionError("invalid retrieval method must be refused")


def test_provenance_ebook_id_mismatch_rejected():
    """Provenance sidecar whose ebook_id disagrees with the filename
    mapping must be refused (defends against pasting the wrong sidecar
    with the right filename)."""
    from scripts import vendor_aeon_lbc1 as V
    from scripts.vendor_aeon_lbc1 import AcquisitionError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        bad = _provenance("pg-0011.txt", override_ebook_id=9999)
        intake = _build_intake(tmp,
                                override_provenance={"pg-0011.txt": bad})
        try:
            V.import_offline_sources(tmp / "AEON-LBC-1", intake)
        except AcquisitionError as e:
            assert e.code == "provenance_ebook_id_mismatch"
        else:
            raise AssertionError("ebook_id mismatch must be refused")


# ---------------------------------------------------------------------------
# §1.9 — Incoming files are read-only (byte-preserved) — vendor writes
#         to package, not back to intake
# ---------------------------------------------------------------------------
def test_intake_files_remain_byte_identical():
    from scripts import vendor_aeon_lbc1 as V
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        intake = _build_intake(tmp)
        pre = {}
        for filename in _TITLES:
            path = intake / "sources" / filename
            pre[filename] = (path.read_bytes(),
                              hashlib.sha256(path.read_bytes()).hexdigest())
        V.import_offline_sources(tmp / "AEON-LBC-1", intake)
        for filename, (before, digest) in pre.items():
            path = intake / "sources" / filename
            after = path.read_bytes()
            assert after == before
            assert hashlib.sha256(after).hexdigest() == digest


# ---------------------------------------------------------------------------
# §4 — Atomic promotion: partial validation failure leaves no package
# ---------------------------------------------------------------------------
def test_partial_import_never_becomes_active_package():
    """If any source fails validation, no source file in the package
    directory should carry the current-run digest — the previous
    package state remains authoritative."""
    from scripts import vendor_aeon_lbc1 as V
    from scripts.vendor_aeon_lbc1 import AcquisitionError
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # First: land a valid package.
        good_intake = _build_intake(tmp)
        pkg = tmp / "AEON-LBC-1"
        V.import_offline_sources(pkg, good_intake)
        digests_before = json.loads(
            (pkg / "ORIGINAL_SOURCE_DIGESTS").read_text())
        # Second: attempt an intake where ONE source is corrupt.
        bad_intake = _build_intake(
            tmp / "second",
            override_bytes={"pg-2701.txt": b"NOT_A_GUTENBERG_FILE"})
        try:
            V.import_offline_sources(pkg, bad_intake)
        except AcquisitionError:
            pass
        else:
            raise AssertionError("bad intake should have raised")
        # Original digests are unchanged.
        digests_after = json.loads(
            (pkg / "ORIGINAL_SOURCE_DIGESTS").read_text())
        assert digests_before == digests_after


# ---------------------------------------------------------------------------
# §8 — Corpus notices do not modify Aeon licensing (re-cited here so a
#       future editor cannot regress it without hitting this file too)
# ---------------------------------------------------------------------------
def test_corpus_notices_still_do_not_touch_aeon_source():
    forbidden = ("Project Gutenberg is a registered trademark",
                  "This eBook is for the use of anyone anywhere")
    for name in ("LICENSE", "LICENSE.md", "LICENSE.txt", "README.md"):
        p = os.path.join(ROOT, name)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", errors="replace") as fh:
            src = fh.read()
        for phrase in forbidden:
            assert phrase not in src


# ---------------------------------------------------------------------------
# §8 — No corpus file enters the installer bundle
# ---------------------------------------------------------------------------
def test_no_incoming_corpus_enters_installer_bundle():
    iss = os.path.join(ROOT, "packaging", "windows", "AeonInstaller.iss")
    if os.path.exists(iss):
        src = open(iss, encoding="utf-8").read()
        for token in ("research-data", "incoming/AEON-LBC-1",
                        "AEON-LBC-1/source"):
            assert token not in src


# ---------------------------------------------------------------------------
# §1 — Network and offline modes are mutually exclusive (CLI)
# ---------------------------------------------------------------------------
def test_cli_refuses_refresh_source_in_offline_mode():
    from scripts import vendor_aeon_lbc1 as V
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        intake = _build_intake(tmp)
        rc = V.main(["--package-root", str(tmp / "AEON-LBC-1"),
                      "--source-dir", str(intake),
                      "--refresh-source"])
        assert rc == 2


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
