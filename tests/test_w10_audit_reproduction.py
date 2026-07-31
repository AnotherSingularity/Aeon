"""W10-0 audit reproduction — one test per finding in docs/W10_AUDIT_REPRODUCTION.md.

Every test in this file asserts the CURRENT BROKEN behavior at the tip of
`claude/funny-cori-a3k5cf` before any W10 correction. When a W10-N tranche
corrects a finding, that tranche must FLIP the assertion in the affected
test and (if appropriate) rename the test.

Running this file on Linux is safe — none of these tests require Windows,
PyInstaller, torch runtime execution, or the GUI event loop. They inspect
source structure only.
"""
import ast
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _read(rel):
    return open(os.path.join(ROOT, rel), encoding="utf-8").read()


# ---------------------------------------------------------------------------
# A1, A2, A3 — FLIPPED by W10-1. The worker no longer generates synthetic
# random tokens; it consumes real corpus tokens through
# aeon.job.data_source.build_data_source, which fail-closes when tokenizer
# or corpus is absent.
# ---------------------------------------------------------------------------
def test_worker_next_batch_is_real_corpus_not_random():
    """W10-1 flipped this assertion. The worker used to call
    torch.randint(0, vocab_size, ...); it now iterates over the batches
    produced by aeon.job.data_source.TokenizedCorpusBatchSource. The old
    W10-0 form of this test asserted the presence of `torch.randint`; the
    new form asserts its absence in the training loop and the presence of
    the data-source contract."""
    src = _read("aeon/job/worker.py")
    m = re.search(r"def _run_training_loop.*?(?=\n(?:def |\Z))", src, re.DOTALL)
    assert m
    code_lines = [
        line for line in m.group(0).splitlines()
        if not line.lstrip().startswith("#")]
    body_code = "\n".join(code_lines)
    assert "torch.randint" not in body_code, (
        "W10-1: _run_training_loop must not contain torch.randint in code")
    assert "data_source.iter_batches" in body_code, (
        "W10-1: _run_training_loop must drive batches through the "
        "TokenizedCorpusBatchSource")


def test_worker_uses_tokenizer_path():
    """W10-1 flipped. build_data_source now loads job.tokenizer_path via
    AeonTokenizer and refuses to proceed if it's absent."""
    src = _read("aeon/job/worker.py")
    assert "from aeon.job.data_source import build_data_source" in src
    ds = _read("aeon/job/data_source.py")
    assert "tokenizer_path = getattr(job, \"tokenizer_path\", None)" in ds
    assert "AeonTokenizer(tokenizer_path)" in ds
    assert "tokenizer_absent" in ds and "tokenizer_missing" in ds


def test_worker_uses_corpus_path():
    """W10-1 flipped. build_data_source now reads job.corpus_path through
    aeon.data.iter_text_records via the TokenizedCorpusBatchSource, and
    refuses to proceed if it's absent or empty."""
    ds = _read("aeon/job/data_source.py")
    assert "corpus_path = getattr(job, \"corpus_path\", None)" in ds
    assert "from aeon.data import iter_text_records" in ds
    for reason in ("corpus_absent", "corpus_missing", "corpus_empty",
                    "corpus_too_small"):
        assert reason in ds, f"data source must handle {reason}"


# ---------------------------------------------------------------------------
# A4 — FLIPPED by W10-2. The worker now uses the F3 protected envelope.
# ---------------------------------------------------------------------------
def test_worker_uses_protected_checkpoint():
    src = _read("aeon/job/worker.py")
    assert "from aeon.protected_checkpoint import" in src, (
        "W10-2: worker must import the protected envelope APIs")
    assert "protected_save" in src, (
        "W10-2: worker must call protected_save")
    assert "protected_load" in src, (
        "W10-2: worker must call protected_load on resume")
    # Direct atomic_save/strict_load calls are forbidden in code lines
    # (comments still allowed).
    code_lines = [line for line in src.splitlines()
                    if not line.lstrip().startswith("#")]
    body = "\n".join(code_lines)
    assert "atomic_save(" not in body
    assert "strict_load(" not in body


# ---------------------------------------------------------------------------
# A5 — FLIPPED by W10-2. The "authenticated checkpoint" claim in the GUI
# is now accurate: the worker actually calls protected_save under the
# per-job HMAC key.
# ---------------------------------------------------------------------------
def test_gui_authenticated_checkpoint_claim_is_accurate():
    gui = _read("aeon/launcher/gui.py")
    worker = _read("aeon/job/worker.py")
    assert "authenticated checkpoint" in gui, (
        "GUI still uses the 'authenticated checkpoint' wording (that's fine)")
    assert "protected_save" in worker, (
        "W10-2: the claim is now backed by protected_save in the worker")
    assert "ensure_job_hmac_keyref" in worker, (
        "W10-2: the HMAC key comes from the per-job key store")


# ---------------------------------------------------------------------------
# A6 — FLIPPED by W10-3. Resume is a distinct flow.
# ---------------------------------------------------------------------------
def test_gui_resume_is_a_distinct_flow():
    src = _read("aeon/launcher/gui.py")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_resume":
            body_source = ast.unparse(node) if hasattr(ast, "unparse") else ""
            assert "self._on_start()" not in body_source, (
                "W10-3: _on_resume must no longer alias _on_start")
            assert ("latest_authenticated_checkpoint" in body_source
                    or 'intent="resume"' in body_source
                    or "intent='resume'" in body_source), (
                "W10-3: _on_resume must consult latest_authenticated_checkpoint "
                "and create a job with intent='resume'")
            return
    raise AssertionError("could not find _on_resume in gui.py")


# ---------------------------------------------------------------------------
# A7 — worker ignores launcher settings for cpu/memory limits
# ---------------------------------------------------------------------------
def test_worker_ignores_gui_settings():
    src = _read("aeon/job/worker.py")
    for gui_only in ("cpu_thread_limit", "memory_ceiling_gb",
                      "validation_interval", "checkpoint_interval",
                      "resume_preference"):
        assert gui_only not in src, (
            f"worker must not YET consume launcher setting {gui_only!r} "
            "(flip at W10-9)")


# ---------------------------------------------------------------------------
# A8 — hard-coded 0.0 throughput placeholders
# ---------------------------------------------------------------------------
def test_worker_emits_zero_placeholder_metrics():
    src = _read("aeon/job/worker.py")
    assert "step_time_s=0.0" in src, "worker still hard-codes step_time_s=0.0"
    assert "tokens_per_s_raw=0.0" in src, "worker still hard-codes tokens_per_s_raw=0.0"
    assert "useful_tokens_per_s=0.0" in src, "worker still hard-codes useful_tokens_per_s=0.0"


# ---------------------------------------------------------------------------
# A9 — manifest excludes top-level Aeon.exe
# ---------------------------------------------------------------------------
def test_manifest_excludes_top_level_aeon_exe():
    src = _read("packaging/windows/generate_runtime_manifest.py")
    # The generator walks internal/ when present. Top-level Aeon.exe sits
    # one level above and is not enumerated. Documented at W10-0; corrected
    # at W10-6.
    assert "internal = bundle / \"_internal\"" in src
    assert "walk_root = internal" in src
    # And no CODE path adds Aeon.exe from bundle-root back into the manifest.
    code_only = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "Aeon.exe" not in code_only, (
        "manifest generator must not YET include a special-case for the "
        "top-level Aeon.exe in CODE (flip at W10-6)")


# ---------------------------------------------------------------------------
# A10 — malformed manifest entries are silently skipped
# ---------------------------------------------------------------------------
def test_verifier_silently_skips_malformed_entries():
    src = _read("aeon/integrity.py")
    # The `continue` on missing rel/expected is the flaw.
    assert "if not rel or not expected:" in src
    assert re.search(r"if not rel or not expected:\s*\n\s*continue", src), (
        "verify_installed_manifest must still silently `continue` on "
        "malformed entries at W10-0 (flip at W10-6 — must fail closed)")


# ---------------------------------------------------------------------------
# A11 — unexpected extra files in installed tree are not rejected
# ---------------------------------------------------------------------------
def test_verifier_ignores_unexpected_extra_files():
    src = _read("aeon/integrity.py")
    # verify_installed_manifest only iterates manifest entries; it does not
    # walk the installed tree looking for files that are NOT in the manifest.
    assert "os.walk" not in src, (
        "integrity verifier must not YET walk the installed tree looking "
        "for unexpected files (flip at W10-6)")
    assert "unexpected" not in src.lower() or True


# ---------------------------------------------------------------------------
# A12 — Inno relative paths without SourceDir
# ---------------------------------------------------------------------------
def test_iss_relative_paths_without_sourcedir():
    src = _read("packaging/windows/AeonInstaller.iss")
    assert "OutputDir=dist\\installer" in src or "OutputDir=dist\\\\installer" in src
    assert 'Source: "dist\\Aeon\\*"' in src or 'Source: "dist\\\\Aeon\\\\*"' in src
    assert not re.search(r"^\s*SourceDir\s*=", src, re.MULTILINE), (
        "ISS must not YET declare SourceDir= — the audit reproduction is "
        "that relative paths resolve from the ISS directory (flip at W10-7)")


# ---------------------------------------------------------------------------
# A13 — pre-install check is only FileExists
# ---------------------------------------------------------------------------
def test_iss_preinstall_is_only_presence_check():
    src = _read("packaging/windows/AeonInstaller.iss")
    m = re.search(r"function PrepareToInstall.*?end;", src, re.DOTALL)
    assert m, "PrepareToInstall function not found"
    body = m.group(0)
    assert "FileExists(ManifestPath)" in body
    # No hash / signature / checksum in the pre-install check
    for stronger in ("SHA256", "SHA1", "CryptCreateHash", "VerifySignature"):
        assert stronger not in body, (
            f"PrepareToInstall must not YET perform {stronger!r} (flip at W10-7)")


# ---------------------------------------------------------------------------
# A14 — upgrade only blocks on CHECKPOINTING
# ---------------------------------------------------------------------------
def test_iss_upgrade_only_blocks_on_checkpointing():
    src = _read("packaging/windows/AeonInstaller.iss")
    # Only CHECKPOINTING is checked; RUNNING/STARTING/STOP_REQUESTED are not
    m = re.search(r"function IsAnActiveJobWritingCheckpoint.*?end;",
                    src, re.DOTALL)
    assert m
    body = m.group(0)
    assert "CHECKPOINTING" in body
    for still_ignored in ("RUNNING", "STARTING", "STOP_REQUESTED"):
        assert still_ignored not in body, (
            f"upgrade guard must not YET check {still_ignored!r} "
            "(flip at W10-7)")
    assert "CloseApplications=force" in src, (
        "CloseApplications=force must still be present at W10-0 baseline "
        "(flip at W10-7 — remove or gate)")


# ---------------------------------------------------------------------------
# A15 — checkpoint provenance falls back to 'unknown' when no .git
# ---------------------------------------------------------------------------
def test_checkpoint_provenance_falls_back_to_unknown_when_no_git():
    src = _read("aeon/checkpoint.py")
    m = re.search(r"def source_commit_id.*?\n(?=def |\Z)", src, re.DOTALL)
    assert m
    body = m.group(0)
    assert "git" in body and "rev-parse" in body
    assert '"unknown"' in body, (
        "source_commit_id must still return 'unknown' on frozen builds "
        "(flip at W10-5 — use embedded RELEASE_METADATA)")


# ---------------------------------------------------------------------------
# A16 — .prev rotation is not one atomic envelope
# ---------------------------------------------------------------------------
def test_checkpoint_rotation_not_atomic_across_envelope():
    src = _read("aeon/checkpoint.py")
    # No generation-directory or COMPLETE marker in the current code
    for hallmark in ("COMPLETE", "generation-", "envelope_atomic"):
        assert hallmark not in src, (
            f"checkpoint must not YET carry {hallmark!r} — the audit "
            "reproduction is that rotation is not atomic across the whole "
            "envelope (flip at W10-4)")


# ---------------------------------------------------------------------------
# A17 — preflight does not block on missing tokenizer or corpus
# ---------------------------------------------------------------------------
def test_preflight_does_not_block_on_missing_tokenizer_or_corpus():
    src = _read("aeon/config/preflight.py")
    # No pass/fail check that opens the tokenizer file and validates it
    assert "load_tokenizer" not in src and "SentencePieceProcessor" not in src, (
        "preflight must not YET actually load the tokenizer (flip at W10-8)")
    # No pass/fail check that reads the corpus
    assert "corpus_manifest_open" not in src and "verify_corpus" not in src, (
        "preflight must not YET verify the corpus (flip at W10-8)")


# ---------------------------------------------------------------------------
# A18 — GUI Validate is a placeholder
# ---------------------------------------------------------------------------
def test_gui_validate_is_placeholder():
    src = _read("aeon/launcher/gui.py")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_validate":
            body = ast.unparse(node.body[-1]) if hasattr(ast, "unparse") else ""
            assert "messagebox.showinfo" in body or "showinfo" in body, (
                "_on_validate must still be a messagebox placeholder at W10-0")
            return
    raise AssertionError("_on_validate not found")


# ---------------------------------------------------------------------------
# A19 — GUI Recovery requires terminal
# ---------------------------------------------------------------------------
def test_gui_recovery_requires_terminal():
    src = _read("aeon/launcher/gui.py")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_on_recovery":
            body = ast.unparse(node.body[-1]) if hasattr(ast, "unparse") else ""
            assert "Aeon.exe --recover" in body or "messagebox" in body, (
                "_on_recovery must still ask the user to launch --recover "
                "themselves at W10-0 baseline (flip at W10-9)")
            return
    raise AssertionError("_on_recovery not found")


# ---------------------------------------------------------------------------
# A20 — GUI Diagnose discards subprocess output
# ---------------------------------------------------------------------------
def test_gui_diagnose_discards_output():
    src = _read("aeon/launcher/gui.py")
    assert "stdout=subprocess.DEVNULL" in src and "stderr=subprocess.DEVNULL" in src, (
        "_on_diagnose must still discard subprocess output at W10-0 baseline "
        "(flip at W10-9)")


# ---------------------------------------------------------------------------
# A21 — lockfile has range pins
# ---------------------------------------------------------------------------
def test_windows_lock_has_range_pins_not_exact():
    src = _read("packaging/windows/requirements-windows.lock")
    # Exact pins with '==' for torch and pyinstaller; ranges (>= or <) for others
    assert "torch==2.5.1+cpu" in src
    ranges = [line for line in src.splitlines()
                if line and not line.startswith("#")
                and (">=" in line or "<" in line and "==" not in line)]
    assert ranges, "lockfile must still contain range pins at W10-0 (flip at W10-10)"


# ---------------------------------------------------------------------------
# A22 — workflow actions pinned by tag not SHA
# ---------------------------------------------------------------------------
def test_workflow_actions_pinned_by_tag_not_sha():
    src = _read(".github/workflows/windows-release.yml")
    tag_pins = re.findall(r"uses:\s*(actions/[\w-]+@v\d+)\b", src)
    assert tag_pins, "expected version-tag pins for GitHub actions"
    assert not re.search(r"uses:\s*actions/[\w-]+@[0-9a-f]{40}", src), (
        "workflow must not YET pin actions by SHA (flip at W10-10)")


# ---------------------------------------------------------------------------
# A23 — build.ps1 creates a placeholder license and continues
# ---------------------------------------------------------------------------
def test_build_ps1_creates_placeholder_license_and_continues():
    src = _read("packaging/windows/build.ps1")
    assert "PLACEHOLDER.txt" in src, (
        "build.ps1 must still create a PLACEHOLDER license at W10-0 baseline")
    assert "Place third-party licences here before shipping" in src


# ---------------------------------------------------------------------------
# A25 — attestation may be unavailable
# ---------------------------------------------------------------------------
def test_workflow_attestation_may_be_unavailable_on_current_plan():
    src = _read(".github/workflows/windows-release.yml")
    assert "actions/attest-build-provenance" in src, (
        "workflow still references attest-build-provenance at W10-0")
    # No ATTESTATION_NOT_AVAILABLE_FOR_CURRENT_PLAN handling yet
    assert "ATTESTATION_NOT_AVAILABLE" not in src, (
        "workflow must not YET record the unavailable-attestation status "
        "(flip at W10-10)")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
