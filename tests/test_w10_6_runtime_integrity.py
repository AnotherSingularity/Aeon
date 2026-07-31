"""W10-6 — full-bundle runtime integrity.

Covers audit findings A9 (top-level Aeon.exe excluded from the manifest),
A10 (malformed entries silently skipped), and A11 (unexpected extra files
in the installed tree not rejected).
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = os.path.join(ROOT, "packaging", "windows")
sys.path.insert(0, ROOT)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_frozen_style_bundle(d):
    """Simulate a PyInstaller 6.x onedir layout inside ``d``:

        d/
            Aeon.exe                 <-- top-level, previously outside the manifest
            python311.dll            <-- top-level
            _internal/
                configs/aeon_smoke.yaml
                torch/_C.pyd
                packaging/windows/    (manifest lands here)
    """
    internal = os.path.join(d, "_internal")
    os.makedirs(os.path.join(internal, "configs"))
    os.makedirs(os.path.join(internal, "torch"))
    os.makedirs(os.path.join(internal, "packaging", "windows"))
    Path(os.path.join(d, "Aeon.exe")).write_bytes(b"MZ_frozen_exe_body")
    Path(os.path.join(d, "python311.dll")).write_bytes(b"MZ_python_dll_body")
    Path(os.path.join(internal, "configs", "aeon_smoke.yaml")).write_text("model: {}\n")
    Path(os.path.join(internal, "torch", "_C.pyd")).write_bytes(b"MZ_torch_c_body")
    # RELEASE.json under packaging/windows
    rel = {"semantic_version": "0.2.3", "source_commit": "abc",
            "build_type": "release", "signed": False}
    rel_path = os.path.join(d, "RELEASE.json")
    with open(rel_path, "w") as fh:
        json.dump(rel, fh)
    return internal, rel_path


def _generate(bundle_dir, rel_path):
    r = subprocess.run(
        [sys.executable,
          os.path.join(PKG, "generate_runtime_manifest.py"),
          "--bundle", bundle_dir, "--release", rel_path],
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    manifest_path = os.path.join(
        bundle_dir, "_internal", "packaging", "windows", "RUNTIME_MANIFEST.json")
    return manifest_path


# ---------------------------------------------------------------------------
# A9: manifest includes top-level Aeon.exe
# ---------------------------------------------------------------------------
def test_manifest_includes_top_level_aeon_exe():
    with tempfile.TemporaryDirectory() as d:
        _build_frozen_style_bundle(d)
        mp = _generate(d, os.path.join(d, "RELEASE.json"))
        m = json.load(open(mp, encoding="utf-8"))
        paths = [f["path"] for f in m["files"]]
        assert "../Aeon.exe" in paths, (
            "W10-6/A9: manifest must include top-level Aeon.exe under "
            "'../Aeon.exe'; got: " + str(paths))
        assert "../python311.dll" in paths


def test_manifest_top_level_entries_carry_scope_flag():
    with tempfile.TemporaryDirectory() as d:
        _build_frozen_style_bundle(d)
        mp = _generate(d, os.path.join(d, "RELEASE.json"))
        m = json.load(open(mp, encoding="utf-8"))
        for f in m["files"]:
            if f["path"].startswith("../"):
                assert f.get("scope") == "top_level", f
            else:
                assert f.get("scope") == "internal", f


def test_manifest_schema_and_trust_root_recorded():
    with tempfile.TemporaryDirectory() as d:
        _build_frozen_style_bundle(d)
        mp = _generate(d, os.path.join(d, "RELEASE.json"))
        m = json.load(open(mp, encoding="utf-8"))
        assert m["manifest_schema_version"] >= 2
        tr = m["trust_root"]
        assert tr["kind"] == "sha256_per_file"
        assert tr["signed_manifest"] is False
        assert tr["adversary_integrity_scope"] == "none"
        assert "full_bundle_including_top_level" in tr["accidental_integrity_scope"]


# ---------------------------------------------------------------------------
# Verifier: fail closed on malformed / unexpected / traversal
# ---------------------------------------------------------------------------
def _patched_verify(bundle_dir):
    """Point installed_resource_root at ``bundle_dir/_internal`` and run
    verify_installed_manifest. Returns (ok, report)."""
    from aeon import integrity
    from aeon import windows_paths as wp
    internal = Path(bundle_dir) / "_internal"
    with mock.patch("aeon.integrity.installed_resource_root",
                     return_value=internal):
        return integrity.verify_installed_manifest()


def test_verifier_detects_tampered_top_level_aeon_exe():
    with tempfile.TemporaryDirectory() as d:
        _build_frozen_style_bundle(d)
        _generate(d, os.path.join(d, "RELEASE.json"))
        ok, report = _patched_verify(d)
        assert ok is True, report
        # Tamper Aeon.exe
        Path(os.path.join(d, "Aeon.exe")).write_bytes(b"MZ_tampered_exe")
        ok2, report2 = _patched_verify(d)
        assert ok2 is False, "W10-6/A9: tampered top-level Aeon.exe must fail verification"
        assert any(x["path"] == "../Aeon.exe" for x in report2["mismatched"]), report2


def test_verifier_fails_closed_on_malformed_entries():
    with tempfile.TemporaryDirectory() as d:
        _build_frozen_style_bundle(d)
        mp = _generate(d, os.path.join(d, "RELEASE.json"))
        m = json.load(open(mp))
        # Insert a malformed entry (missing sha256)
        m["files"].append({"path": "configs/broken.yaml", "bytes": 0})
        with open(mp, "w") as fh:
            json.dump(m, fh)
        ok, report = _patched_verify(d)
        assert ok is False, "W10-6/A10: malformed entry must fail closed"
        assert report["malformed"], report


def test_verifier_rejects_path_traversal_entries():
    with tempfile.TemporaryDirectory() as d:
        _build_frozen_style_bundle(d)
        mp = _generate(d, os.path.join(d, "RELEASE.json"))
        m = json.load(open(mp))
        # Path traversal attempt
        m["files"].append({"path": "../../etc/passwd", "bytes": 1,
                            "sha256": "0" * 64, "scope": "internal"})
        with open(mp, "w") as fh:
            json.dump(m, fh)
        ok, report = _patched_verify(d)
        assert ok is False
        assert (any(x.get("entry", {}).get("path") == "../../etc/passwd"
                     for x in report["malformed"])
                 or any("../etc" in p or p.startswith("../..") for p in report["missing"])), report


def test_verifier_rejects_unexpected_extra_executable():
    with tempfile.TemporaryDirectory() as d:
        internal, rel_path = _build_frozen_style_bundle(d)
        _generate(d, rel_path)
        # Add an unexpected .exe INSIDE _internal — the audit's A11 case.
        Path(os.path.join(internal, "malware.exe")).write_bytes(b"MZ_extra")
        ok, report = _patched_verify(d)
        assert ok is False, "W10-6/A11: unexpected extra .exe must fail verification"
        assert "malware.exe" in report["unexpected"], report


def test_verifier_rejects_unexpected_extra_dll():
    with tempfile.TemporaryDirectory() as d:
        internal, rel_path = _build_frozen_style_bundle(d)
        _generate(d, rel_path)
        Path(os.path.join(internal, "shady.dll")).write_bytes(b"MZ_dll")
        ok, report = _patched_verify(d)
        assert ok is False
        assert "shady.dll" in report["unexpected"]


def test_verifier_report_includes_trust_root_and_schema():
    with tempfile.TemporaryDirectory() as d:
        _build_frozen_style_bundle(d)
        _generate(d, os.path.join(d, "RELEASE.json"))
        ok, report = _patched_verify(d)
        assert ok
        assert report["manifest_schema_version"] >= 2
        assert report["trust_root"]["kind"] == "sha256_per_file"


def test_integrity_source_no_silent_continue_on_missing_hash():
    """Source-level check: the old silent `continue` on missing path/sha256
    is gone."""
    src = open(os.path.join(ROOT, "aeon/integrity.py"), encoding="utf-8").read()
    assert "malformed.append" in src, (
        "integrity verifier must record malformed entries, not silently skip")
    # And "if not rel or not expected: continue" is gone from CODE lines.
    import re
    code_lines = [line for line in src.splitlines()
                    if not line.lstrip().startswith("#")]
    body = "\n".join(code_lines)
    assert re.search(r"if not rel or not expected:\s*\n\s*continue", body) is None, (
        "W10-6/A10: the silent continue on malformed entries must be gone")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
