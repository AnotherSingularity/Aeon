"""W5-W7 packaging-source tests.

Cannot BUILD Aeon.exe on Linux, but we can verify:
  * The PyInstaller .spec exists and lists the required Aeon submodules.
  * The runtime hook file exists and sets AEON_DATA_DIR on Windows.
  * The Inno Setup .iss exists and enforces the required behaviour
    (per-user install, no admin, preserves user data on uninstall, refuses
    upgrade during checkpointing).
  * generate_runtime_manifest.py produces a valid manifest that
    aeon.integrity.verify_installed_manifest can consume.
  * release_metadata.py never writes secrets.
  * sign.ps1 references only environment variables for credentials — never
    hard-codes any secret.
  * The packaging tree contains no PFX / .key / .pem / other key artefacts.
  * requirements-windows.lock pins CPU torch (no CUDA).
"""
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


PKG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "packaging", "windows")


# ---- W5: PyInstaller spec + hook -------------------------------------------
def test_spec_file_exists_and_lists_aeon_hidden_imports():
    p = os.path.join(PKG, "Aeon.spec")
    assert os.path.exists(p), p
    src = open(p).read()
    # A representative sample of packages that MUST ship
    required = ["aeon.entry", "aeon.hybrid", "aeon.recursion",
                "aeon.transformer", "aeon.substrate.matrix_cell",
                "aeon.checkpoint", "aeon.protected_checkpoint",
                "aeon.launcher.gui", "aeon.job.worker",
                "aeon.config.schema", "aeon.integrity", "aeon.evidence"]
    missing = [m for m in required if m not in src]
    assert not missing, f"spec missing Aeon submodules: {missing}"


def test_spec_uses_onedir_and_windowed():
    src = open(os.path.join(PKG, "Aeon.spec")).read()
    assert "exclude_binaries=True" in src, "onedir requires exclude_binaries=True"
    assert "console=False" in src, "windowed subsystem requires console=False"
    assert "upx=False" in src, "§W5 forbids UPX"


def test_spec_excludes_test_dir_and_cuda():
    src = open(os.path.join(PKG, "Aeon.spec")).read()
    for banned in ("'tests'", "torch.cuda", "torch.backends.cudnn"):
        assert banned in src, f"spec must exclude {banned}"


def test_runtime_hook_exists_and_sets_aeon_data_dir_on_windows():
    p = os.path.join(PKG, "runtime_hook.py")
    assert os.path.exists(p)
    src = open(p).read()
    assert "AEON_DATA_DIR" in src
    assert "LOCALAPPDATA" in src


def test_requirements_lock_pins_cpu_torch_and_no_cuda():
    p = os.path.join(PKG, "requirements-windows.lock")
    src = open(p).read()
    assert "torch==2.5.1+cpu" in src, "must pin CPU torch"
    # No CUDA extras
    for cuda_hint in ("+cu", "cu124", "cu121", "cu118"):
        assert cuda_hint not in src, f"unexpected CUDA reference in lock: {cuda_hint}"


# ---- W5: manifest generator ------------------------------------------------
def test_manifest_generator_produces_valid_manifest_verifier_accepts():
    """Run the manifest generator on a small tree; feed the result to
    aeon.integrity.verify_installed_manifest — must pass. Then tamper — must fail."""
    from aeon.integrity import verify_installed_manifest, MANIFEST_RELATIVE
    from aeon import windows_paths as wp
    with tempfile.TemporaryDirectory() as d:
        # Build a fake bundle
        os.makedirs(os.path.join(d, "packaging", "windows"))
        os.makedirs(os.path.join(d, "configs"))
        for name in ("Aeon.exe", "configs/aeon_smoke_e5.yaml", "python311.dll"):
            path = os.path.join(d, name)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as fh:
                fh.write(b"data-" + name.encode())
        # Write RELEASE.json
        rel = os.path.join(PKG, "RELEASE.json")
        release = {"semantic_version": "0.2.3", "source_commit": "abc123",
                    "build_type": "development"}
        rel_path = os.path.join(d, "packaging", "windows", "RELEASE.json")
        with open(rel_path, "w") as fh:
            json.dump(release, fh)
        # Run the generator as a subprocess
        r = subprocess.run(
            [sys.executable, os.path.join(PKG, "generate_runtime_manifest.py"),
             "--bundle", d, "--release", rel_path],
            capture_output=True, text=True, timeout=30)
        assert r.returncode == 0, r.stderr
        assert os.path.exists(os.path.join(d, MANIFEST_RELATIVE))
        # Verify
        import unittest.mock as mock
        with mock.patch.object(wp, "installed_resource_root",
                                 return_value=type("P", (), {"__truediv__": lambda self, x: os.path.join(d, x), "exists": lambda self: True})()):
            # Simpler: pretend the whole path is what integrity expects
            from pathlib import Path
            with mock.patch("aeon.integrity.installed_resource_root",
                             return_value=Path(d)):
                ok, report = verify_installed_manifest()
                assert ok, report
                assert report["files_ok"] >= 3
                # Tamper one file
                with open(os.path.join(d, "Aeon.exe"), "ab") as fh:
                    fh.write(b"tamper")
                ok2, report2 = verify_installed_manifest()
                assert not ok2
                assert any(x["path"] == "Aeon.exe" for x in report2["mismatched"])


# ---- W7: release_metadata never writes secrets -----------------------------
def test_release_metadata_never_writes_secrets():
    src = open(os.path.join(PKG, "release_metadata.py")).read()
    # No env-var reads for secret material
    for secret in ("AEON_SIGN_CERT_PASS", "SIGN_CERT_PASS", "PFX_PASSWORD",
                    "CERTIFICATE_PASSWORD", "SIGNING_PASSWORD"):
        assert secret not in src, f"release_metadata.py references secret {secret}"
    # The output structure has --signed only as a boolean flag
    assert "--signed" in src
    assert "action=\"store_true\"" in src


# ---- W7: sign.ps1 uses env vars, no hard-coded secrets ---------------------
def test_sign_ps1_uses_env_vars_only():
    p = os.path.join(PKG, "sign.ps1")
    src = open(p).read()
    # Required env references
    for envkey in ("AEON_SIGN_CERT_PFX", "AEON_SIGN_CERT_PASS",
                    "AEON_SIGNTOOL_PATH"):
        assert envkey in src, f"sign.ps1 must reference {envkey}"
    # No hardcoded password / cert
    forbidden_hints = ["PFX_PASSWORD=", "'-p'",
                        "-Password ", "\"MyPassword", "cert_password"]
    for f in forbidden_hints:
        assert f not in src, f"sign.ps1 has hardcoded credential hint: {f}"


# ---- W6: Inno Setup script enforces required behaviour ---------------------
def test_iss_declares_per_user_no_admin():
    src = open(os.path.join(PKG, "AeonInstaller.iss")).read()
    assert "PrivilegesRequired=lowest" in src
    # {localappdata}\Programs\<app> — <app> may be the preprocessor token {#AppName}
    assert ("{localappdata}\\Programs\\Aeon" in src
            or "{localappdata}\\Programs\\{#AppName}" in src)


def test_iss_preserves_user_data_by_default():
    src = open(os.path.join(PKG, "AeonInstaller.iss")).read()
    # Uninstall prompt must be MB_DEFBUTTON2 so YES (delete) isn't default
    assert "MB_DEFBUTTON2" in src, "uninstall data-deletion must NOT be the default action"
    assert "DelTree" in src, "uninstall must OFFER data purge behind a confirmation"


def test_iss_refuses_upgrade_during_checkpointing():
    src = open(os.path.join(PKG, "AeonInstaller.iss")).read()
    assert "CHECKPOINTING" in src, "installer must refuse upgrade while checkpointing"


# ---- Packaging tree contains no key/cert artefacts --------------------------
def test_no_signing_material_committed_anywhere():
    """Scan packaging/windows/ for any *.pfx / *.key / *.pem / *.p12 / *.crt file."""
    forbidden_ext = (".pfx", ".key", ".pem", ".p12", ".crt", ".pass")
    offenders = []
    for root, _, files in os.walk(PKG):
        for f in files:
            if f.lower().endswith(forbidden_ext):
                offenders.append(os.path.join(root, f))
    assert not offenders, offenders


# ---- Documentation hygiene: packaging scripts contain no absolute host paths
def test_packaging_scripts_contain_no_absolute_user_paths():
    import re
    offenders = []
    for name in ("Aeon.spec", "runtime_hook.py", "build.ps1",
                  "build_installer.ps1", "sign.ps1",
                  "generate_runtime_manifest.py",
                  "release_metadata.py", "AeonInstaller.iss",
                  "file_version_info.txt"):
        p = os.path.join(PKG, name)
        if not os.path.exists(p): continue
        text = open(p).read()
        for m in re.finditer(r"/home/[a-z]+|/Users/[a-z]+", text):
            offenders.append(f"{name}: {m.group(0)}")
    assert not offenders, offenders


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
