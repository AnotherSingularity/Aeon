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
    """W10-2 flipped this. W10-4 refactored save via generation_save,
    which internally calls protected_save. Either mention counts; direct
    atomic_save / strict_load calls remain forbidden."""
    src = _read("aeon/job/worker.py")
    gen = _read("aeon/job/generation.py")
    assert "from aeon.protected_checkpoint import" in src, (
        "worker must still import protected-load APIs")
    assert ("protected_save" in src or "generation_save" in src), (
        "W10-2+W10-4: worker save path must go through protected_save "
        "(directly or via generation_save)")
    assert "protected_save" in gen, "generation_save must call protected_save"
    assert "protected_load" in src
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
    gen = _read("aeon/job/generation.py")
    assert "authenticated checkpoint" in gui, (
        "GUI still uses the 'authenticated checkpoint' wording (that's fine)")
    # protected_save may live in generation_save now (W10-4 refactor).
    assert ("protected_save" in worker or "generation_save" in worker), (
        "W10-2+W10-4: the claim is backed by protected_save via "
        "generation_save in the worker")
    assert "protected_save" in gen, (
        "W10-4: generation_save must call protected_save")
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
# A9 — manifest excludes top-level Aeon.exe  [CORRECTED W10-6]
# ---------------------------------------------------------------------------
def test_manifest_excludes_top_level_aeon_exe():
    src = _read("packaging/windows/generate_runtime_manifest.py")
    # W10-6 flip: the generator now enumerates top-level bundle files with a
    # "../" prefix and scope="top_level". A9 CORRECTED.
    code_only = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert 'rel = f"../{name}"' in code_only, (
        "W10-6/A9: manifest generator must enumerate top-level bundle files "
        "with a '../' prefix")
    assert '"scope": "top_level"' in code_only, (
        "W10-6/A9: top-level entries must carry scope='top_level'")


# ---------------------------------------------------------------------------
# A10 — malformed manifest entries are silently skipped  [CORRECTED W10-6]
# ---------------------------------------------------------------------------
def test_verifier_silently_skips_malformed_entries():
    src = _read("aeon/integrity.py")
    # W10-6 flip: the silent `continue` on malformed entries has been
    # replaced with a `malformed.append(...)` that fails verification.
    code_only = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert re.search(r"if not rel or not expected:\s*\n\s*continue", code_only) is None, (
        "W10-6/A10: silent continue on malformed entries must be gone")
    assert "malformed.append" in code_only, (
        "W10-6/A10: verifier must record malformed entries so verification fails")


# ---------------------------------------------------------------------------
# A11 — unexpected extra files in installed tree are not rejected  [CORRECTED W10-6]
# ---------------------------------------------------------------------------
def test_verifier_ignores_unexpected_extra_files():
    src = _read("aeon/integrity.py")
    # W10-6 flip: verify_installed_manifest now walks the installed tree
    # and rejects unlisted .exe/.dll/.pyd/etc. files.
    code_only = "\n".join(
        line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert "os.walk" in code_only, (
        "W10-6/A11: verifier must walk the installed tree to detect "
        "unexpected files")
    assert "unexpected" in code_only, (
        "W10-6/A11: verifier must populate an 'unexpected' report list")


# ---------------------------------------------------------------------------
# A12 — Inno relative paths without SourceDir  [CORRECTED W10-7]
# ---------------------------------------------------------------------------
def test_iss_relative_paths_without_sourcedir():
    src = _read("packaging/windows/AeonInstaller.iss")
    # Strip ; comments (Inno) and // comments before searching for the
    # SourceDir declaration so a corrective docstring can't spoof the check.
    non_comment = "\n".join(l for l in src.splitlines()
                              if not l.lstrip().startswith(";")
                              and not l.lstrip().startswith("//"))
    assert re.search(r"^\s*SourceDir\s*=", non_comment, re.MULTILINE), (
        "W10-7/A12: [Setup] must declare SourceDir= so [Files] Source paths "
        "resolve from a stable base")


# ---------------------------------------------------------------------------
# A13 — pre-install check is only FileExists  [CORRECTED W10-7]
# ---------------------------------------------------------------------------
def test_iss_preinstall_is_only_presence_check():
    src = _read("packaging/windows/AeonInstaller.iss")
    m = re.search(r"function PrepareToInstall.*?(?=\nfunction |\Z)", src, re.DOTALL)
    assert m, "PrepareToInstall function not found"
    body = m.group(0)
    assert "GetSHA256OfFile" in body, (
        "W10-7/A13: PrepareToInstall must compute a SHA-256 of the manifest, "
        "not just FileExists")
    assert "RUNTIME_MANIFEST.sha256" in body, (
        "W10-7/A13: PrepareToInstall must read the sha256 sidecar written "
        "by generate_runtime_manifest.py")


# ---------------------------------------------------------------------------
# A14 — upgrade only blocks on CHECKPOINTING  [CORRECTED W10-7]
# ---------------------------------------------------------------------------
def test_iss_upgrade_only_blocks_on_checkpointing():
    src = _read("packaging/windows/AeonInstaller.iss")
    m = re.search(r"function IsAnActiveJob(?:WritingCheckpoint)?\(\).*?(?=\nfunction |\Z)",
                    src, re.DOTALL)
    assert m
    body = m.group(0)
    for required in ("CHECKPOINTING", "RUNNING", "STARTING", "STOP_REQUESTED"):
        assert required in body, (
            f"W10-7/A14: upgrade guard must block on {required!r}")
    # Strip comments before checking the removal of CloseApplications=force
    non_comment = "\n".join(l for l in src.splitlines()
                              if not l.lstrip().startswith(";")
                              and not l.lstrip().startswith("//"))
    assert "CloseApplications=force" not in non_comment, (
        "W10-7/A14: CloseApplications=force must be removed")


# ---------------------------------------------------------------------------
# A15 — FLIPPED by W10-5. Frozen mode consults RELEASE_METADATA and raises
# SourceCommitUnavailable rather than returning 'unknown'.
# ---------------------------------------------------------------------------
def test_checkpoint_provenance_no_unknown_fallback_when_frozen():
    src = _read("aeon/checkpoint.py")
    assert "class SourceCommitUnavailable" in src
    m = re.search(r"def source_commit_id.*?(?=\nclass |\ndef |\Z)",
                    src, re.DOTALL)
    assert m
    body = m.group(0)
    assert "is_frozen()" in body
    assert "RELEASE_METADATA" in body
    assert "SourceCommitUnavailable" in body


# ---------------------------------------------------------------------------
# A16 — FLIPPED by W10-4. Atomic per-generation checkpoint chain with
# COMPLETE markers now guarantees a crash cannot leave a half-rotated
# envelope discoverable.
# ---------------------------------------------------------------------------
def test_checkpoint_rotation_is_atomic_across_envelope():
    """W10-4 flipped. The atomic-generation implementation lives in
    aeon/job/generation.py rather than aeon/checkpoint.py — the E3 file
    stays unchanged so inherited tests keep passing — and the worker's
    _save_checkpoint calls generation_save which writes a COMPLETE
    marker LAST and then atomically renames the .tmp directory."""
    gen_src = _read("aeon/job/generation.py")
    assert "COMPLETE_MARKER" in gen_src and 'STATE_FILENAME = "state.pt"' in gen_src
    assert "def generation_save" in gen_src
    assert "os.rename(str(tmp), str(target))" in gen_src, (
        "W10-4: promotion must be a single atomic rename")
    assert "LATEST_POINTER" in gen_src and "os.replace" in gen_src, (
        "W10-4: latest-authorized.txt update must be atomic")
    worker = _read("aeon/job/worker.py")
    assert "from aeon.job.generation import" in worker
    assert "generation_save(job.checkpoint_dir" in worker
    assert "discard_incomplete(job.checkpoint_dir)" in worker


# ---------------------------------------------------------------------------
# A17 — preflight does not block on missing tokenizer or corpus  [CORRECTED W10-8]
# ---------------------------------------------------------------------------
def test_preflight_does_not_block_on_missing_tokenizer_or_corpus():
    src = _read("aeon/config/preflight.py")
    # W10-8 flip: frozen mode now fails closed on missing tokenizer/corpus.
    assert "_is_frozen" in src, (
        "W10-8/A17: preflight must distinguish frozen from source mode")
    assert "unusable_status" in src, (
        "W10-8/A17: preflight must select fail-vs-warn on frozen mode")
    assert "iter_text_records" in src or "_corpus_read_fails" in src, (
        "W10-8/A17: preflight must actually attempt to READ the corpus, "
        "not merely check for existence")


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
