# W10-0 — Audit Reproduction

Each row below reproduces one finding from the external audit against the
current branch tip. Every claim carries a `file:line` citation into the
authoritative source at commit `96c017a` (before any W10 change). Where a
finding is machine-testable, the "reproduction test" column names a case in
`tests/test_w10_audit_reproduction.py` that asserts the CURRENT (broken)
behavior; a later W10-N checkpoint that fixes the behavior flips that
assertion.

Nothing in W10-0 CHANGES the runtime. W10-0 only stops overstating what it
guarantees. Every row below explicitly identifies:

* status = **REPRODUCED** — audit finding confirmed against source
* status = **REPRODUCED PARTIAL** — audit finding correct but with nuance
* owning W10-N checkpoint — the tranche that will actually correct it

| # | audit finding | source citation | status | reproduction test | owned by |
|---|---|---|---|---|---|
| A1 | GUI worker generates random token batches instead of training on English | `aeon/job/worker.py:145–150` (`return torch.randint(0, tcfg_model.vocab_size, ...)`) | **CORRECTED (W10-1)** | flipped to `test_worker_next_batch_is_real_corpus_not_random` | W10-1 |
| A2 | Configured tokenizer path is not opened or used by the training loop | `aeon/job/worker.py:73–150` — `_run_training_loop` never opens `job.tokenizer_path`; the whole loop constructs batches from `torch.randint` | **CORRECTED (W10-1)** | flipped to `test_worker_uses_tokenizer_path`; positive coverage in `tests/test_w10_1_real_corpus.py` (15 checks) | W10-1 |
| A3 | Configured corpus path is not opened or used by the training loop | `aeon/job/worker.py:73–150` — `job.corpus_path` is never read anywhere in `_run_training_loop` | **CORRECTED (W10-1)** | flipped to `test_worker_uses_corpus_path`; positive coverage in `tests/test_w10_1_real_corpus.py` | W10-1 |
| A4 | Worker uses ordinary atomic_save/strict_load, NOT the protected checkpoint envelope | `aeon/job/worker.py:84` (`from aeon.checkpoint import atomic_save, strict_load, build_metadata, latest_checkpoint`); `aeon/job/worker.py:212–223` (`_save_checkpoint` calls `atomic_save(...)` — no HMAC, no anti-rollback, no `protected_save`) | **CORRECTED (W10-2)** | flipped to `test_worker_uses_protected_checkpoint`; full round-trip coverage in `tests/test_w10_2_protected_checkpoint.py` (13 checks); trust posture documented in `aeon/job/key_store.py` (per-job HMAC = development integrity, not adversary integrity — that trust root arrives at W10-6) | W10-2 |
| A5 | GUI labels ordinary checkpoint behavior as "authenticated checkpoint" | `aeon/launcher/gui.py:329–332` (Safe stop message: "…save an authenticated checkpoint…"); `aeon/launcher/gui.py:344–347` (Validate message: "…latest authenticated checkpoint…") | **CORRECTED (W10-2)** | flipped to `test_gui_authenticated_checkpoint_claim_is_accurate` — the GUI wording is unchanged, but the underlying worker now actually produces the F3 HMAC/anti-rollback envelope | W10-2 |
| A6 | Resume Latest is aliased to Start | `aeon/launcher/gui.py:320–324` (`def _on_resume(self): self._on_start()`) | **CORRECTED (W10-3)** | flipped to `test_gui_resume_is_a_distinct_flow`; full coverage in `tests/test_w10_3_distinct_flows.py` (14 checks) covering Job.intent field, `aeon/launcher/resume.py` enumeration, Start-refuses-active-chain gate, and distinct audit-event kinds `start_new_training`/`resume_latest`/`recovery_authorized` | W10-3 |
| A7 | Configured checkpoint_interval, validation_interval, cpu_thread_limit, memory_ceiling_gb, resume_preference are recorded but not enforced by the worker | `aeon/job/worker.py:73–207` — the worker reads only `tcfg["ckpt_every"]`, `tcfg["log_every"]`, `tcfg["max_steps"]`, `tcfg["batch_size"]`, `tcfg["seed"]`, `tcfg["grad_clip"]`, `tcfg.get("aux_gate_penalty")`, `tcfg.get("resume")`, and `tcfg.get("sample_every")`. No enforcement of cpu-thread-limit, memory-ceiling, or the launcher-side validation_interval | REPRODUCED PARTIAL — the launcher does write these into `user_config.json` but the worker never reads them | `test_worker_ignores_gui_settings` | W10-9 |
| A8 | Runtime throughput metrics are hard-coded zero placeholders | `aeon/job/worker.py:191` (`step_time_s=0.0, tokens_per_s_raw=0.0, useful_tokens_per_s=0.0`) | REPRODUCED | `test_worker_emits_zero_placeholder_metrics` | W10-9 |
| A9 | Runtime manifest excludes the top-level Aeon.exe | `packaging/windows/generate_runtime_manifest.py:53–65` — walks `bundle/_internal/` (or `bundle/` on fallback) and computes paths relative to the walk root. `Aeon.exe` sits at `dist/Aeon/Aeon.exe`, ONE LEVEL ABOVE `_internal/`, so it is never enumerated when `_internal/` is present (the PyInstaller 6.x layout). | CORRECTED (W10-6) | `test_manifest_excludes_top_level_aeon_exe` (flipped) + `tests/test_w10_6_runtime_integrity.py::test_manifest_includes_top_level_aeon_exe`, `::test_manifest_top_level_entries_carry_scope_flag`, `::test_verifier_detects_tampered_top_level_aeon_exe` | W10-6 |
| A10 | Malformed manifest entries are silently skipped instead of failing closed | `aeon/integrity.py:55–56` (`if not rel or not expected: continue`) | CORRECTED (W10-6) | `test_verifier_silently_skips_malformed_entries` (flipped) + `tests/test_w10_6_runtime_integrity.py::test_verifier_fails_closed_on_malformed_entries`, `::test_verifier_rejects_path_traversal_entries` | W10-6 |
| A11 | Unexpected added executables/DLLs in the installed tree are not rejected | `aeon/integrity.py:42–72` — verifier only iterates the manifest and checks per-file presence + hash; it never walks the installed tree to detect files ADDED beyond the manifest | CORRECTED (W10-6) | `test_verifier_ignores_unexpected_extra_files` (flipped) + `tests/test_w10_6_runtime_integrity.py::test_verifier_rejects_unexpected_extra_executable`, `::test_verifier_rejects_unexpected_extra_dll` | W10-6 |
| A12 | Inno Setup `[Files] Source:` and `OutputDir=` use bare relative paths with no `SourceDir=` set | `packaging/windows/AeonInstaller.iss:34` (`OutputDir=dist\installer`), `packaging/windows/AeonInstaller.iss:60` (`Source: "dist\Aeon\*"`), and no `SourceDir=` anywhere in the file. Inno resolves relative `[Files] Source:` from the ISS-file directory, so ISCC would look at `packaging\windows\dist\Aeon` — the wrong tree | CORRECTED (W10-7) | `test_iss_relative_paths_without_sourcedir` (flipped) + `tests/test_w10_7_installer_correctness.py::test_iss_declares_sourcedir_two_levels_up`, `::test_iss_files_source_still_uses_dist_aeon` | W10-7 |
| A13 | Installer pre-install check verifies presence, not payload content | `packaging/windows/AeonInstaller.iss:120–134` — `PrepareToInstall` calls `FileExists(ManifestPath)`, no hash check. `ManifestPath` is `{src}\dist\Aeon\_internal\packaging\windows\RUNTIME_MANIFEST.json`, where `{src}` is the directory containing `AeonSetup.exe` at install time — the original build tree does not exist beside a distributed installer | CORRECTED (W10-7) | `test_iss_preinstall_is_only_presence_check` (flipped) + `tests/test_w10_7_installer_correctness.py::test_iss_preinstall_verifies_manifest_sha256`, `::test_generator_emits_sha256_sidecar`, `::test_generator_does_not_list_sidecar_in_manifest` | W10-7 |
| A14 | Installer can proceed while a live worker is in RUNNING/STARTING/STOP_REQUESTED — only CHECKPOINTING blocks | `packaging/windows/AeonInstaller.iss:158` (`if Pos('CHECKPOINTING', Status) > 0 then Result := True`); `packaging/windows/AeonInstaller.iss:41` (`CloseApplications=force`) | CORRECTED (W10-7) | `test_iss_upgrade_only_blocks_on_checkpointing` (flipped) + `tests/test_w10_7_installer_correctness.py::test_iss_upgrade_guard_covers_all_live_states`, `::test_iss_removes_close_applications_force`, `::test_initializesetup_calls_expanded_guard` | W10-7 |
| A15 | Frozen checkpoint provenance falls back to `unknown` | `aeon/checkpoint.py:54–61` (`source_commit_id()` runs `git rev-parse HEAD` and returns `"unknown"` on any exception — including the frozen case where no `.git` exists); `aeon/checkpoint.py:92` (`build_metadata` calls `source_commit_id()` unconditionally) | **CORRECTED (W10-5)** | flipped to `test_checkpoint_provenance_no_unknown_fallback_when_frozen`; 7 W10-5 checks in `tests/test_w10_5_frozen_provenance.py`. Frozen mode reads `aeon.version.RELEASE_METADATA["source_commit"]` and raises `SourceCommitUnavailable` if the metadata is missing or reports `"unknown"`. `"unknown"` is now returnable ONLY in source-tree mode when git AND RELEASE_METADATA are both absent. | W10-5 |
| A16 | .prev payload and its verification metadata are not rotated as one recoverable generation | `aeon/checkpoint.py::atomic_save` writes a `.prev` payload alongside the new payload, but the sha256 sidecar and metadata JSON are not held in the same atomic transaction — a crash between the payload write and the sidecar write leaves an unauthenticated `.prev`. `aeon/protected_checkpoint.py` has the same multi-file layout without a completion marker across payload + metadata + MAC + provenance. | **CORRECTED (W10-4)** | flipped to `test_checkpoint_rotation_is_atomic_across_envelope`; full coverage in `tests/test_w10_4_atomic_generation.py` (12 checks) exercising the `generation-<step>.tmp/` → `COMPLETE` → `generation-<step>/` promotion, `latest-authorized.txt` atomic pointer, incomplete-generation cleanup, and previous-generation recovery after tampering the current one | W10-4 |
| A17 | Frozen preflight can return READY without a usable tokenizer or corpus | `aeon/config/preflight.py` — none of the 17 checks require a tokenizer file to load or a corpus to open. Missing tokenizer/corpus paths generate WARN, not BLOCKED. | REPRODUCED | `test_preflight_does_not_block_on_missing_tokenizer_or_corpus` | W10-8 |
| A18 | GUI Validate is a modal message box, no validation runs | `aeon/launcher/gui.py:344–347` — `_on_validate` only calls `messagebox.showinfo`; it does not construct a bounded validator or run one. | REPRODUCED | `test_gui_validate_is_placeholder` | W10-9 |
| A19 | GUI Recovery instructs the user to open a terminal | `aeon/launcher/gui.py:359–362` — `_on_recovery` displays a message asking the user to launch `Aeon.exe --recover <request.json>` themselves | REPRODUCED | `test_gui_recovery_requires_terminal` | W10-9 |
| A20 | GUI Diagnose discards subprocess output | `aeon/launcher/gui.py:354–355` — `subprocess.Popen([..., "--diagnose", ck], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)` | REPRODUCED | `test_gui_diagnose_discards_output` | W10-9 |
| A21 | Dependency lock uses version ranges, not exact pins with hashes | `packaging/windows/requirements-windows.lock` — pins `torch==2.5.1+cpu` and `pyinstaller==6.11.1` exactly, but declares ranges for `safetensors>=0.4,<1.0`, `sentencepiece>=0.2,<1.0`, `pyyaml>=6.0`, `numpy>=1.24,<2.0`, `pyinstaller-hooks-contrib>=2024.1`. No hashes | REPRODUCED | `test_windows_lock_has_range_pins_not_exact` | W10-10 |
| A22 | Workflow actions are pinned by version tag, not immutable SHA | `.github/workflows/windows-release.yml:51` (`actions/checkout@v4`), `:162` (`actions/setup-python@v5`), `:404,412` (`actions/upload-artifact@v4`), `:436,475,481,554` (`actions/download-artifact@v4`, `actions/upload-artifact@v4`), `:452` (`actions/attest-build-provenance@v1`) | REPRODUCED | `test_workflow_actions_pinned_by_tag_not_sha` | W10-10 |
| A23 | Placeholder license file may ship in place of real notices | `packaging/windows/build.ps1:73–78` — if `packaging/windows/licenses/` does not exist, `build.ps1` creates it with a `PLACEHOLDER.txt` reading "Place third-party licences here before shipping" and continues the build without failing | REPRODUCED | `test_build_ps1_creates_placeholder_license_and_continues` | W10-10 |
| A24 | W9 DoD marks unsupported behavior as MET | `docs/W9_DEFINITION_OF_DONE.md` row 5 (safe-stop → "atomic authenticated checkpoint at boundary" — MET), row 6 (single-instance lock), row 29 (atomic writes for checkpoint), item "K=16", item "HMAC-authenticated envelope", etc. — the row-5 MET is invalidated by A4; the "HMAC envelope in the GUI worker" claim in §2 (Inherited invariants preserved) is invalidated by A4. | REPRODUCED PARTIAL — the underlying `aeon.protected_checkpoint` module DOES implement HMAC + anti-rollback; the falsehood is that the GUI worker uses it. W9 §2 conflates "the invariant exists in the module" with "the GUI worker enforces it". | W10-0 (withdrawal) + W10-2 (correction) |
| A25 | Attestation may be unavailable for private repos on the current plan | `.github/workflows/windows-release.yml:451–454` — uses `actions/attest-build-provenance@v1`. GitHub documents artifact attestations for private repos as requiring GitHub Enterprise Cloud. This repo is private (per API `"private":true`). | REPRODUCED (documentation-side) | `test_workflow_attestation_may_be_unavailable_on_current_plan` | W10-10 |

## Findings the audit raised that require nuance

| # | audit finding | current state | disposition |
|---|---|---|---|
| N1 | "OS and architecture are always marked pass" in preflight | Preflight does check `sys.platform` and `sys.maxsize` heuristics but only WARNs, never BLOCKS | REPRODUCED PARTIAL. Correct under W10-8. |
| N2 | "Corpus provenance is largely path existence" | The corpus_manifest module in `aeon/corpus_manifest.py` DOES implement schema + provenance checks; the launcher just doesn't invoke it before Start | REPRODUCED. Correct under W10-1 (wire the loader) + W10-8 (block on missing). |
| N3 | "torch.compile / TorchScript deprecation warnings during build" | Emitted by PyTorch 2.5.1 during PyInstaller's analysis phase; not a defect in Aeon | ACKNOWLEDGED, not owned by W10. |
| N4 | "the manifest is not itself rooted in a trusted signature" | Correct as stated; the current manifest is a bare JSON file beside the files it verifies | REPRODUCED. Correct under W10-6 (trusted root — signed manifest OR embedded digest). |

## Cross-reference: follow-on program (post-W10)

Once W10-11 closes, a separate research and runtime program — the
**L-series (Latent Bypass and Hidden-State Efficiency Upgrade)** — begins
from the certified W10 final commit. The L-series proves or disproves,
with causal and cost-adjusted evidence, whether Aeon's hidden Recursion
state provides a lower-cost computational route around barriers in visible
local computation.

The L-series is mechanically gated. The gate lives at
`docs/W10_CLOSURE_STATE.json` and is enforced by
`tests/test_l_series_prerequisite_lock.py`. No L-series runtime file
(reaction-coordinate estimator, barrier registry, intervention harness,
Bayes-factor analysis, matched-control experiment) can land in the repo
while any W10 tranche's `closed` field is still `false`.

The full L-series directive text is preserved at
`docs/directives/L_SERIES_LATENT_BYPASS_AND_HIDDEN_STATE_EFFICIENCY_UPGRADE.md`
so W10 work can proceed with clear sight of the downstream requirement
without executing it prematurely.

## Cross-reference: what W10-0 does NOT do

W10-0 does not fix ANY of the reproduced findings. Its only responsibilities:

1. Reproduce the findings against source (this document).
2. Withdraw the affected claims from W9 and W8 (banners + row downgrades).
3. Stop the Windows CI loop from burning runner minutes against a build that
   we now know would produce a misleading installer even if it succeeded.
4. Lock the reproductions in `tests/test_w10_audit_reproduction.py` so
   future work cannot silently regress back to the pre-W10 misstatements.

The corrections themselves are owned by W10-1 through W10-11 as listed in
the "owned by" column above.
