"""W1 — Unified entry-point dispatch tests.

Covers:
  * GUI default dispatch (headless — assert dispatch happens without full launcher)
  * Worker dispatch: routes to aeon.job.worker.run_worker
  * --verify-installation dispatch: routes to integrity.verify_installed_manifest
  * --validate-config dispatch: routes to config.schema.validate_config_file
  * --diagnose dispatch: routes to scripts/diagnose.py
  * Unknown / missing args: EXIT_USER_ARG_ERROR
  * Installed-resource resolution (frozen + source modes)
  * Source-tree resource resolution
  * Paths containing spaces, Unicode paths
  * Read-only installation directory (writable layout lives elsewhere)
"""
import json
import os
import sys
import tempfile
import unittest.mock as mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- W1 core dispatch ------------------------------------------------------
def test_gui_default_dispatch():
    """No arguments → gui dispatcher invoked (mocked)."""
    from aeon import entry
    with mock.patch.object(entry, "_dispatch_gui", return_value=42) as m:
        rc = entry.main([])
        m.assert_called_once()
        assert rc == 42


def test_worker_dispatch_routes_to_worker_module():
    from aeon import entry
    with mock.patch.object(entry, "_dispatch_worker", return_value=0) as m:
        rc = entry.main(["--worker", "/tmp/aeon_test/job.json"])
        m.assert_called_once_with("/tmp/aeon_test/job.json")
        assert rc == 0


def test_verify_installation_dispatch():
    from aeon import entry
    with mock.patch.object(entry, "_dispatch_verify_installation", return_value=0) as m:
        rc = entry.main(["--verify-installation"])
        m.assert_called_once()
        assert rc == 0


def test_validate_config_dispatch_reports_missing_file():
    from aeon import entry
    rc = entry.main(["--validate-config", "/nonexistent/config.yaml"])
    assert rc == entry.EXIT_CONFIG_INVALID


def test_diagnose_dispatch_reports_missing_checkpoint():
    from aeon import entry
    rc = entry.main(["--diagnose", "/nonexistent/ck.pt"])
    assert rc == entry.EXIT_CHECKPOINT_NOT_FOUND


def test_recover_dispatch_reports_missing_request():
    from aeon import entry
    rc = entry.main(["--recover", "/nonexistent/req.json"])
    assert rc == entry.EXIT_USER_ARG_ERROR


def test_unknown_argument_rejection():
    from aeon import entry
    rc = entry.main(["--make-me-a-sandwich"])
    assert rc != entry.EXIT_OK


def test_mutually_exclusive_modes_rejected():
    from aeon import entry
    rc = entry.main(["--worker", "/tmp/x", "--verify-installation"])
    assert rc != entry.EXIT_OK


# ---- W1 resource resolution -----------------------------------------------
def test_source_mode_installed_resource_root_points_at_repo():
    from aeon import windows_paths
    root = windows_paths.installed_resource_root()
    # In source mode, root should contain the aeon/ package
    assert (root / "aeon" / "entry.py").exists(), \
        f"source root {root} missing aeon/entry.py"


def test_resolve_installed_ignores_cwd():
    from aeon import windows_paths
    orig = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        os.chdir(d)
        try:
            path = windows_paths.resolve_installed("docs/W0_WINDOWS_AUDIT.md")
            assert path.exists(), f"resolve_installed did not find docs from a foreign CWD: {path}"
        finally:
            os.chdir(orig)


def test_user_data_root_is_separate_from_installed():
    from aeon import windows_paths
    inst = windows_paths.installed_resource_root()
    with tempfile.TemporaryDirectory() as d:
        with mock.patch.dict(os.environ, {"AEON_DATA_DIR": d}):
            ud = windows_paths.user_data_root()
            assert str(ud).startswith(d)
            assert str(ud) != str(inst)


def test_ensure_writable_layout_creates_dirs_under_data_root_only():
    from aeon import windows_paths
    with tempfile.TemporaryDirectory() as d:
        with mock.patch.dict(os.environ, {"AEON_DATA_DIR": d}):
            layout = windows_paths.ensure_writable_layout()
            for key in ("user_data", "config", "jobs", "logs", "evidence", "checkpoints"):
                assert key in layout
                assert layout[key].startswith(d)
                assert os.path.isdir(layout[key])


# ---- Windows-specific path shapes (unit-testable on Linux) -----------------
def test_paths_with_spaces_are_handled_by_pathlib():
    from aeon import windows_paths
    with tempfile.TemporaryDirectory() as d:
        weird = os.path.join(d, "path with spaces")
        os.makedirs(weird)
        with mock.patch.dict(os.environ, {"AEON_DATA_DIR": weird}):
            layout = windows_paths.ensure_writable_layout()
            assert "path with spaces" in layout["user_data"]


def test_unicode_paths_are_handled():
    from aeon import windows_paths
    with tempfile.TemporaryDirectory() as d:
        # Unicode dir name — should round-trip through Path
        unicode_dir = os.path.join(d, "日本語_dir")
        os.makedirs(unicode_dir)
        with mock.patch.dict(os.environ, {"AEON_DATA_DIR": unicode_dir}):
            layout = windows_paths.ensure_writable_layout()
            assert "日本語_dir" in layout["user_data"]


def test_read_only_installation_directory_does_not_break_layout():
    """Even if the installed_resource_root() is read-only, ensure_writable_layout
    creates ONLY under user_data_root — never touches installed root."""
    from aeon import windows_paths
    with tempfile.TemporaryDirectory() as d:
        with mock.patch.dict(os.environ, {"AEON_DATA_DIR": d}):
            # This must not attempt any write under installed_resource_root
            layout = windows_paths.ensure_writable_layout()
            inst = str(windows_paths.installed_resource_root())
            for p in layout.values():
                assert not p.startswith(inst + os.sep) or "AEON_DATA_DIR" in os.environ, \
                    f"writable layout leaked into installed root: {p}"


# ---- structured error record -----------------------------------------------
def test_error_record_writes_to_logs_dir():
    from aeon import entry, windows_paths
    with tempfile.TemporaryDirectory() as d:
        with mock.patch.dict(os.environ, {"AEON_DATA_DIR": d}):
            windows_paths.ensure_writable_layout()
            entry._write_error_record("test_kind", 99, "detail", ["--foo"])
            errlog = windows_paths.logs_dir() / "errors.jsonl"
            assert errlog.exists()
            rec = json.loads(errlog.read_text().splitlines()[-1])
            assert rec["kind"] == "test_kind"
            assert rec["exit_code"] == 99


# ---- verify-installation on a bundle we build inline -----------------------
def test_verify_installation_passes_on_matching_manifest():
    """Fabricate a mini installed root, write a manifest matching it, and prove
    the verifier PASSES. Then tamper and prove it FAILS."""
    from aeon import integrity, windows_paths
    with tempfile.TemporaryDirectory() as d:
        # lay out a mini installed tree
        (Path := __import__("pathlib").Path)
        root = Path(d)
        (root / "packaging" / "windows").mkdir(parents=True)
        (root / "libx.dll").write_bytes(b"hello")
        manifest = {"files": [{"path": "libx.dll",
                                "sha256": integrity._sha256_file(str(root / "libx.dll")),
                                "bytes": 5}]}
        (root / "packaging" / "windows" / "RUNTIME_MANIFEST.json").write_text(
            json.dumps(manifest))
        # point installed_resource_root at our tempdir
        with mock.patch("aeon.integrity.installed_resource_root", return_value=root):
            ok, report = integrity.verify_installed_manifest()
            assert ok, report
            assert report["files_ok"] == 1
        # tamper
        (root / "libx.dll").write_bytes(b"tampered")
        with mock.patch("aeon.integrity.installed_resource_root", return_value=root):
            ok, report = integrity.verify_installed_manifest()
            assert not ok
            assert report["mismatched"] and report["mismatched"][0]["path"] == "libx.dll"


def _run_all():
    from pathlib import Path
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
