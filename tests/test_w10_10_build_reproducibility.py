"""W10-10 — build reproducibility, real licenses, attestation handling.

Covers audit findings A21 / A22 / A23 / A25:

    A21 requirements-windows.lock must use exact "==" pins for every
        runtime and build-tool dependency.
    A22 GitHub Actions references must be pinned to immutable 40-char
        commit SHAs, not moving version tags.
    A23 build.ps1 must refuse to build without real third-party
        licences — no PLACEHOLDER fallback.
    A25 windows-release.yml must handle the documented private-repo
        attestation-unavailable path without failing the job.
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def _read(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


# ---------------------------------------------------------------------------
# A21 — exact pins only
# ---------------------------------------------------------------------------
def test_lockfile_has_no_range_pins():
    src = _read("packaging/windows/requirements-windows.lock")
    for line in src.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        # Every non-comment line must be an exact pin.
        assert "==" in s, (
            f"W10-10/A21: non-exact pin in lockfile: {s!r}")
        assert ">=" not in s and "<" not in s, (
            f"W10-10/A21: range operator in lockfile: {s!r}")


def test_lockfile_pins_every_runtime_dep():
    src = _read("packaging/windows/requirements-windows.lock")
    for req in ("torch", "safetensors", "sentencepiece", "pyyaml",
                  "numpy", "pyinstaller", "pyinstaller-hooks-contrib"):
        assert re.search(rf"^{re.escape(req)}==", src, re.MULTILINE), (
            f"W10-10/A21: lockfile missing exact pin for {req}")


# ---------------------------------------------------------------------------
# A22 — SHA-pinned actions
# ---------------------------------------------------------------------------
def test_workflow_actions_pinned_by_sha():
    for wf in ("windows-release.yml", "windows-certification.yml"):
        src = _read(f".github/workflows/{wf}")
        tag_pins = re.findall(r"uses:\s*(actions/[\w-]+@v\d+)\b", src)
        assert not tag_pins, (
            f"W10-10/A22: {wf} still pins actions by tag: {tag_pins}")
        sha_pins = re.findall(r"uses:\s*actions/[\w-]+@[0-9a-f]{40}", src)
        assert sha_pins, (
            f"W10-10/A22: {wf} has no SHA-pinned action references")


def test_workflow_documents_action_versions_in_comments():
    """Each SHA-pinned uses: line should carry a `# vX.Y.Z` comment so a
    human reader can identify the version without a git lookup."""
    src = _read(".github/workflows/windows-release.yml")
    for m in re.finditer(r"uses:\s*(actions/[\w-]+)@([0-9a-f]{40})(.*)$",
                          src, re.MULTILINE):
        action, sha, tail = m.group(1), m.group(2), m.group(3)
        assert re.search(r"#\s*v\d+", tail), (
            f"W10-10/A22: {action}@{sha} lacks a version-tag comment")


# ---------------------------------------------------------------------------
# A23 — real licenses required
# ---------------------------------------------------------------------------
def test_build_ps1_refuses_missing_licenses_dir():
    src = _read("packaging/windows/build.ps1")
    assert "throw" in src, "build.ps1 must throw when licenses are missing"
    assert re.search(r"licences directory missing", src), (
        "W10-10/A23: build.ps1 must throw when licences/ is missing")


def test_build_ps1_refuses_placeholder_only_licenses_dir():
    src = _read("packaging/windows/build.ps1")
    # New guard checks that the directory has real content beyond
    # PLACEHOLDER.txt; assert the phrasing.
    assert "only contains PLACEHOLDER.txt" in src, (
        "W10-10/A23: build.ps1 must refuse a placeholder-only licences dir")


def test_build_ps1_lists_required_dep_licenses():
    src = _read("packaging/windows/build.ps1")
    for dep in ("torch", "pyinstaller", "sentencepiece", "safetensors",
                  "numpy", "pyyaml"):
        assert re.search(rf"'{re.escape(dep)}'", src) or dep in src, (
            f"W10-10/A23: build.ps1 must require a license for {dep}")


def test_build_ps1_does_not_write_placeholder():
    """The old flow wrote 'Place third-party licences here before shipping.'
    to a PLACEHOLDER.txt. That Set-Content must be gone; the phrase is
    allowed in error messages / documentation comments only."""
    src = _read("packaging/windows/build.ps1")
    assert not re.search(
        r"Set-Content -Path \(Join-Path \$Licenses 'PLACEHOLDER\.txt'\)", src), (
        "W10-10/A23: build.ps1 must not YET write PLACEHOLDER.txt")


# ---------------------------------------------------------------------------
# A25 — attestation availability
# ---------------------------------------------------------------------------
def test_workflow_records_attestation_availability():
    src = _read(".github/workflows/windows-release.yml")
    assert "ATTESTATION_NOT_AVAILABLE_FOR_CURRENT_PLAN" in src, (
        "W10-10/A25: workflow must record the unavailable-attestation status")
    # The attest step must not fail the whole job on that condition.
    m = re.search(
        r"Attest build provenance for AeonSetup\.exe.*?uses: actions/attest",
        src, re.DOTALL)
    assert m, "attest step not found"
    ctx = src[m.start():m.start() + 700]
    assert "continue-on-error: true" in ctx, (
        "W10-10/A25: attest step must have continue-on-error: true")


def test_workflow_attestation_status_is_recorded_in_env():
    src = _read(".github/workflows/windows-release.yml")
    assert "attestation_status=ATTESTED" in src
    assert "steps.attest.outcome" in src or "steps.attest" in src


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
