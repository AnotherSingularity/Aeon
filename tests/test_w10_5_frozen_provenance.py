"""W10-5 — frozen release provenance.

Every test verifies the audit-finding A15 correction: when Aeon is running
as a frozen application (``sys.frozen == True``), ``source_commit_id()``
must NOT fall back to ``"unknown"`` — it must consult
``aeon.version.RELEASE_METADATA`` and raise ``SourceCommitUnavailable`` if
the metadata is absent or itself reports unknown. In source-tree mode, git
remains authoritative.
"""
import os
import sys
import subprocess
import tempfile
import unittest.mock as mock
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def test_source_mode_returns_git_head_when_available():
    from aeon.checkpoint import source_commit_id
    from aeon import windows_paths
    with mock.patch.object(windows_paths, "is_frozen", return_value=False):
        rev = source_commit_id()
    # This test runs inside a real git checkout — we expect a 40-char sha
    # unless git isn't installed, in which case we accept RELEASE_METADATA's
    # value or "unknown" as documented fallback.
    assert isinstance(rev, str) and rev != ""
    if rev != "unknown":
        assert len(rev) >= 7  # short-sha or full


def test_frozen_mode_reads_release_metadata():
    """In frozen mode with a valid RELEASE_METADATA source_commit,
    source_commit_id returns that value."""
    from aeon.checkpoint import source_commit_id
    from aeon import version, windows_paths
    with mock.patch.object(windows_paths, "is_frozen", return_value=True):
        with mock.patch.dict(version.RELEASE_METADATA,
                              {"source_commit": "abc123def456"}):
            assert source_commit_id() == "abc123def456"


def test_frozen_mode_raises_when_metadata_unknown():
    """Audit A15 correction: frozen mode never returns 'unknown'."""
    from aeon.checkpoint import source_commit_id, SourceCommitUnavailable
    from aeon import version, windows_paths
    with mock.patch.object(windows_paths, "is_frozen", return_value=True):
        with mock.patch.dict(version.RELEASE_METADATA,
                              {"source_commit": "unknown"}):
            try:
                source_commit_id()
                raise AssertionError("expected SourceCommitUnavailable")
            except SourceCommitUnavailable:
                pass


def test_frozen_mode_raises_when_metadata_missing():
    from aeon.checkpoint import source_commit_id, SourceCommitUnavailable
    from aeon import version, windows_paths
    with mock.patch.object(windows_paths, "is_frozen", return_value=True):
        # Simulate the source_commit key being entirely absent.
        with mock.patch.dict(version.RELEASE_METADATA, {}, clear=True):
            try:
                source_commit_id()
                raise AssertionError("expected SourceCommitUnavailable")
            except SourceCommitUnavailable:
                pass


def test_source_mode_falls_back_to_release_metadata_when_git_absent():
    from aeon.checkpoint import source_commit_id
    from aeon import version, windows_paths
    with mock.patch.object(windows_paths, "is_frozen", return_value=False):
        # Force the git subprocess to raise
        with mock.patch("aeon.checkpoint.subprocess.run",
                          side_effect=FileNotFoundError("git not on PATH")):
            with mock.patch.dict(version.RELEASE_METADATA,
                                  {"source_commit": "abc123def456"}):
                assert source_commit_id() == "abc123def456"


def test_source_mode_returns_unknown_when_everything_absent():
    from aeon.checkpoint import source_commit_id
    from aeon import version, windows_paths
    with mock.patch.object(windows_paths, "is_frozen", return_value=False):
        with mock.patch("aeon.checkpoint.subprocess.run",
                          side_effect=FileNotFoundError()):
            with mock.patch.dict(version.RELEASE_METADATA,
                                  {"source_commit": "unknown"}):
                # A dev checkout without git installed: 'unknown' is
                # documented as the legitimate result. This is the ONLY
                # remaining path that can return 'unknown'.
                assert source_commit_id() == "unknown"


def test_no_unconditional_git_rev_parse_when_frozen():
    """Source-scan check: the git subprocess call must be inside the
    non-frozen branch, not at the top level of source_commit_id."""
    import re
    src = open(os.path.join(ROOT, "aeon/checkpoint.py"), encoding="utf-8").read()
    m = re.search(r"def source_commit_id.*?(?=\n(?:def |class |\Z))", src, re.DOTALL)
    assert m
    body = m.group(0)
    # The git subprocess call must appear AFTER a frozen check.
    frozen_idx = body.find("is_frozen()")
    git_call_idx = body.find('"git", "rev-parse", "HEAD"')
    assert frozen_idx >= 0 and git_call_idx > frozen_idx, (
        "source_commit_id must consult is_frozen() BEFORE the git subprocess call")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
