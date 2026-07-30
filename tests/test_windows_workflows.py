"""Structural validation of the Windows Tier A / Tier B workflows.

Runs on Linux (no PyYAML required — we use structural regex checks). The
tests below encode the SPECIFIC properties the release directive requires,
so that a future YAML edit that weakens them fails here first.

Covered:
  * Tier A pinned image (windows-2022 — not the moving windows-latest).
  * Concurrency group present, cancel-in-progress: false.
  * Explicit permissions, contents: read (least privilege for the root
    workflow; per-job elevation only where signing/attest requires it).
  * Fixed timeout on every job (no `timeout-minutes` = default 6h).
  * Tier B runs-on requires ALL FOUR labels (self-hosted, windows, x64,
    aeon-certification) — an arbitrary self-hosted runner cannot satisfy it.
  * Signing job is opt-in (workflow_dispatch input) OR tag-triggered, uses
    a protected environment, references credentials only through env vars
    from secrets.
  * No inline plaintext credentials (PFX/PASS/KEY) anywhere in either
    workflow.
  * No use of `${{ secrets.* }}` in shell/pwsh `run:` bodies (secrets must
    reach the script via `env:`, so PowerShell / bash logging never
    interpolates them).
"""
import os
import re


HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
TIER_A = os.path.join(ROOT, ".github", "workflows", "windows-release.yml")
TIER_B = os.path.join(ROOT, ".github", "workflows", "windows-certification.yml")


def _load(path):
    assert os.path.exists(path), f"missing workflow: {path}"
    return open(path, encoding="utf-8").read()


# ---- Tier A ---------------------------------------------------------------

def test_tier_a_exists():
    assert os.path.exists(TIER_A), TIER_A


def test_tier_a_pins_windows_image_not_moving_label():
    src = _load(TIER_A)
    assert "runs-on: windows-2022" in src, "Tier A must pin windows-2022"
    assert "windows-latest" not in src, "Tier A must not use the moving windows-latest label"


def test_tier_a_has_concurrency_and_no_cancel():
    src = _load(TIER_A)
    assert re.search(r"^concurrency:\s*$", src, re.MULTILINE), "concurrency block absent"
    assert "cancel-in-progress: false" in src, "release must not cancel in-progress runs"


def test_tier_a_root_permissions_are_least_privilege():
    src = _load(TIER_A)
    # Root workflow permission: contents: read only.
    m = re.search(r"^permissions:\s*\n\s+contents:\s+read\s*$",
                   src, re.MULTILINE)
    assert m, "root `permissions: contents: read` not asserted"


def test_tier_a_every_job_has_a_timeout():
    src = _load(TIER_A)
    # Every `runs-on:` line must be paired with a `timeout-minutes:` within
    # the same job block (heuristic: check that count matches).
    runs_on = re.findall(r"^\s+runs-on:", src, re.MULTILINE)
    timeouts = re.findall(r"^\s+timeout-minutes:\s*\d+", src, re.MULTILINE)
    assert len(runs_on) == len(timeouts), (
        f"jobs without timeout-minutes: runs_on={len(runs_on)} timeouts={len(timeouts)}")


def test_tier_a_signing_is_opt_in_or_tag_triggered():
    src = _load(TIER_A)
    # The sign job must NOT run on every workflow_dispatch or every push.
    assert "windows-release-signing" in src, "sign job must use protected env"
    m = re.search(r"job:\s*sign\b|^\s+sign:\s*$", src, re.MULTILINE)
    # sign: job header
    assert re.search(r"^\s+sign:\s*$", src, re.MULTILINE), "sign job block missing"
    # sign job's if-condition includes a tag/opt-in check
    assert re.search(r"inputs\.sign\s*==\s*true", src), "sign job must be gated on inputs.sign"
    assert "refs/tags/v" in src, "sign job must also trigger on version tags"


def test_tier_a_no_inline_plaintext_credentials():
    src = _load(TIER_A)
    for bad in ("password", "PFX_PASSWORD=", "-Password ", "AEON_SIGN_CERT_PASS="):
        # Case-insensitive scan, but tolerate the env-var *name* declarations.
        for m in re.finditer(re.escape(bad), src, re.IGNORECASE):
            ctx = src[max(0, m.start() - 40): m.end() + 40]
            # env-var references like ${{ secrets.AEON_SIGN_CERT_PASS }} are fine
            if "secrets." in ctx or "AEON_SIGN_CERT_PASS: " in ctx:
                continue
            # sign.ps1 documentation reference is fine too
            if "sign.ps1" in ctx:
                continue
            raise AssertionError(f"suspect literal near: {ctx!r}")


def test_tier_a_secrets_referenced_only_via_env():
    """Secrets MUST NOT be interpolated directly into run: bodies — they
    reach shells via `env:` blocks, so PowerShell / bash echo can't leak
    them."""
    src = _load(TIER_A)
    # Find every ${{ secrets.* }} occurrence and confirm its context is an
    # env: block, not a run: body.
    for m in re.finditer(r"\$\{\{\s*secrets\.[A-Z0-9_]+\s*\}\}", src):
        # Walk back to the last `run:` or `env:` header before this match.
        head = src[:m.start()]
        run_pos = head.rfind("run: ")
        env_pos = head.rfind("env:")
        # env: must appear more recently than run: — i.e., the secret is in
        # an env: block.
        assert env_pos > run_pos, (
            f"secret interpolated outside env: block at offset {m.start()}: "
            f"{src[max(0, m.start()-40): m.end()+40]!r}")


def test_tier_a_uploads_a_named_installer_artifact():
    src = _load(TIER_A)
    assert "aeon-windows-tier-a-" in src, "Tier A must upload a named installer artefact"
    assert "AeonSetup.exe" in src, "Tier A must produce/copy AeonSetup.exe"
    assert "runtime-manifest" in src.lower() or "RUNTIME_MANIFEST.json" in src


# ---- Tier B ---------------------------------------------------------------

def test_tier_b_exists():
    assert os.path.exists(TIER_B), TIER_B


def test_tier_b_requires_all_four_labels():
    src = _load(TIER_B)
    for lbl in ("self-hosted", "windows", "x64", "aeon-certification"):
        assert re.search(rf"^\s*-\s+{re.escape(lbl)}\s*$", src, re.MULTILINE), (
            f"Tier B must require label {lbl!r}")
    # Also assert: never a bare `runs-on: self-hosted` (which would allow
    # any self-hosted runner). Tier B must be a list of labels.
    assert not re.search(r"runs-on:\s*self-hosted\s*$", src, re.MULTILINE)


def test_tier_b_downloads_tier_a_artifact_and_verifies_sha():
    src = _load(TIER_B)
    assert "actions/download-artifact" in src
    assert "expected_installer_sha256" in src
    assert "SHA-256 mismatch" in src or "SHA256 mismatch" in src or "sha_mismatch" in src


def test_tier_b_refuses_admin_or_system_runner():
    src = _load(TIER_B)
    assert "IsInRole" in src, "Tier B must check IsInRole for administrator"
    assert "Tier B runner must NOT be admin" in src


def test_tier_b_has_manual_signoff_gate():
    src = _load(TIER_B)
    # The workflow must enforce that every interactive check is signed off
    # before Tier B evidence is written.
    assert "manual-signoff.json" in src
    assert "interactive certification incomplete" in src.lower() or \
           "not signed off" in src.lower()


def test_tier_b_no_plaintext_secrets():
    src = _load(TIER_B)
    for m in re.finditer(r"\$\{\{\s*secrets\.[A-Z0-9_]+\s*\}\}", src):
        head = src[:m.start()]
        run_pos = head.rfind("run: ")
        env_pos = head.rfind("env:")
        with_pos = head.rfind("with:")
        # Tier B uses secrets only inside `with:` blocks (github-token) or
        # `env:`; both come AFTER any preceding run:.
        assert env_pos > run_pos or with_pos > run_pos, (
            f"secret interpolated outside env:/with: at offset {m.start()}")


# ---- Both -----------------------------------------------------------------

def test_no_workflow_uses_windows_latest():
    for path in (TIER_A, TIER_B):
        src = _load(path)
        assert "windows-latest" not in src, (
            f"{path}: must pin an explicit runner image, not windows-latest")


def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
