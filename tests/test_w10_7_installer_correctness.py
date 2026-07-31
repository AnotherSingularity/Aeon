"""W10-7 — Inno Setup script correctness + payload verification + upgrade guard.

Covers audit findings:
    A12 — [Files] Source paths need a stable base (SourceDir=).
    A13 — pre-install check was FileExists only; must also verify manifest
          SHA-256 against a sidecar written at build time.
    A14 — upgrade guard only blocked on CHECKPOINTING and used
          CloseApplications=force. Must block on RUNNING / STARTING /
          STOP_REQUESTED / CHECKPOINTING, and must not force-close.
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PKG = os.path.join(ROOT, "packaging", "windows")
ISS = os.path.join(PKG, "AeonInstaller.iss")


def _iss():
    return open(ISS, encoding="utf-8").read()


def _iss_code_only():
    src = _iss()
    return "\n".join(l for l in src.splitlines()
                       if not l.lstrip().startswith(";") and not l.lstrip().startswith("//"))


# ---------------------------------------------------------------------------
# A12: SourceDir pins the base for relative [Files] entries
# ---------------------------------------------------------------------------
def test_iss_declares_sourcedir_two_levels_up():
    src = _iss_code_only()
    m = re.search(r"^\s*SourceDir\s*=\s*(.+?)\s*$", src, re.MULTILINE)
    assert m, "W10-7/A12: [Setup] must declare SourceDir="
    val = m.group(1).strip()
    # The .iss lives at packaging\windows\AeonInstaller.iss, repo root is
    # ..\.. from there. Accept both single- and double-backslash forms.
    assert val in (r"..\..", r"..\\..\\", "../..", ".."), (
        f"W10-7/A12: SourceDir must climb two levels to the repo root; got {val!r}")


def test_iss_files_source_still_uses_dist_aeon():
    src = _iss_code_only()
    assert 'Source: "dist\\Aeon\\*"' in src, (
        "[Files] Source: should still be dist\\Aeon\\* relative to SourceDir")


# ---------------------------------------------------------------------------
# A13: pre-install verifies manifest payload, not just presence
# ---------------------------------------------------------------------------
def test_iss_preinstall_verifies_manifest_sha256():
    src = _iss()
    m = re.search(r"function PrepareToInstall.*?(?=\nfunction |\Z)", src, re.DOTALL)
    assert m, "PrepareToInstall function not found"
    body = m.group(0)
    assert "GetSHA256OfFile" in body, (
        "W10-7/A13: PrepareToInstall must compute a SHA-256 of the manifest")
    assert "RUNTIME_MANIFEST.sha256" in body, (
        "W10-7/A13: PrepareToInstall must read the sha256 sidecar")
    assert "Refusing install" in body or "FAILED" in body, (
        "W10-7/A13: PrepareToInstall must return a non-empty String on mismatch "
        "(Inno treats non-empty as a fatal pre-install error)")


def test_generator_emits_sha256_sidecar():
    """The generator now writes RUNTIME_MANIFEST.sha256 next to the manifest,
    holding the hex SHA-256 of the manifest file. The Inno pre-install check
    reads it via LoadStringFromFile + GetSHA256OfFile."""
    with tempfile.TemporaryDirectory() as d:
        internal = Path(d) / "_internal"
        (internal / "configs").mkdir(parents=True)
        (internal / "configs" / "aeon_smoke.yaml").write_text("model: {}\n")
        Path(d, "Aeon.exe").write_bytes(b"MZ_stub")
        rel = {"semantic_version": "0.2.3", "source_commit": "abc",
                "build_type": "release", "signed": False}
        rel_path = os.path.join(d, "RELEASE.json")
        with open(rel_path, "w") as fh:
            json.dump(rel, fh)
        r = subprocess.run(
            [sys.executable,
              os.path.join(PKG, "generate_runtime_manifest.py"),
              "--bundle", d, "--release", rel_path],
            capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, r.stderr
        m_path = internal / "packaging" / "windows" / "RUNTIME_MANIFEST.json"
        s_path = internal / "packaging" / "windows" / "RUNTIME_MANIFEST.sha256"
        assert m_path.exists()
        assert s_path.exists(), "W10-7/A13: sidecar RUNTIME_MANIFEST.sha256 must exist"
        expected = hashlib.sha256(m_path.read_bytes()).hexdigest()
        actual = s_path.read_text().strip()
        assert actual == expected, (
            f"sidecar content must be the SHA-256 of the manifest; "
            f"got {actual!r} expected {expected!r}")


def test_generator_does_not_list_sidecar_in_manifest():
    """The sidecar itself must not be a manifest entry (circular reference)."""
    with tempfile.TemporaryDirectory() as d:
        internal = Path(d) / "_internal"
        (internal / "configs").mkdir(parents=True)
        (internal / "configs" / "aeon_smoke.yaml").write_text("model: {}\n")
        Path(d, "Aeon.exe").write_bytes(b"MZ_stub")
        rel_path = os.path.join(d, "RELEASE.json")
        with open(rel_path, "w") as fh:
            json.dump({"semantic_version": "0.2.3", "source_commit": "abc",
                        "build_type": "release", "signed": False}, fh)
        subprocess.run(
            [sys.executable,
              os.path.join(PKG, "generate_runtime_manifest.py"),
              "--bundle", d, "--release", rel_path],
            capture_output=True, text=True, timeout=60, check=True)
        m = json.load(open(internal / "packaging" / "windows" / "RUNTIME_MANIFEST.json"))
        for f in m["files"]:
            assert f["path"] != "packaging/windows/RUNTIME_MANIFEST.sha256", (
                "sidecar must not be a manifest entry")


# ---------------------------------------------------------------------------
# A14: upgrade guard covers RUNNING/STARTING/STOP_REQUESTED/CHECKPOINTING
# and CloseApplications=force is gone
# ---------------------------------------------------------------------------
def test_iss_upgrade_guard_covers_all_live_states():
    src = _iss()
    m = re.search(r"function IsAnActiveJob(?:WritingCheckpoint)?\(\).*?(?=\nfunction |\Z)",
                    src, re.DOTALL)
    assert m, "upgrade-guard function not found"
    body = m.group(0)
    for state in ("CHECKPOINTING", "RUNNING", "STARTING", "STOP_REQUESTED"):
        assert state in body, (
            f"W10-7/A14: upgrade guard must block on {state!r}")


def test_iss_removes_close_applications_force():
    # Strip ; and // comments so a corrective docstring mentioning the old
    # value doesn't trip the assertion.
    src = _iss()
    non_comment = "\n".join(l for l in src.splitlines()
                              if not l.lstrip().startswith(";")
                              and not l.lstrip().startswith("//"))
    assert "CloseApplications=force" not in non_comment, (
        "W10-7/A14: CloseApplications=force must be removed — it would kill a "
        "live worker mid-write")
    # And explicit CloseApplications=no is preferred over silent omission
    assert re.search(r"^\s*CloseApplications\s*=\s*no\s*$", non_comment, re.MULTILINE), (
        "W10-7/A14: prefer explicit CloseApplications=no")


def test_initializesetup_calls_expanded_guard():
    src = _iss()
    m = re.search(r"function InitializeSetup.*?end;", src, re.DOTALL)
    assert m
    body = m.group(0)
    # Either the renamed IsAnActiveJob() or backward-compat legacy — both OK
    assert re.search(r"IsAnActiveJob(?:WritingCheckpoint)?\(\)", body), (
        "InitializeSetup must call the upgrade guard")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
