# Desktop Release Candidate — Aeon Desktop 7M Research Preview

**Status:** **STATE B — Functional desktop with genuine Windows-packaging blocker.**

The chat runtime, chat controller, release bundle, streaming, Stop /
New Session / Clear Conversation / Diagnostics, session isolation,
cancellation, and shutdown all work end-to-end under the current
CPU + Linux container. Every generated token executes through the
authentic `aeon.hybrid.HybridModel` — no fallback, no cloud, no mock.

Windows PyInstaller freeze + Inno installer + clean-Windows-machine
acceptance require a Windows runner and are not verifiable from this
Linux container. That is the only gate that separates this build
from FUNCTIONAL_RELEASE_CANDIDATE.

---

## 1. What works, verified

| Requirement                                    | Result | Test                                                              |
| ---------------------------------------------- | ------ | ----------------------------------------------------------------- |
| Inference-only export (state_dict, no optim)   | PASS   | `test_export_is_inference_only`                                   |
| Export is byte-identical to source P2          | PASS   | `test_export_matches_source_checkpoint_bytewise`                  |
| Release bundle has every required field        | PASS   | `test_release_manifest_binds_required_fields`                     |
| Loads via `torch.load(weights_only=True)`      | PASS   | `test_model_export_loads_with_weights_only_true`                  |
| Runtime state machine rejects impossible edges | PASS   | `test_state_machine_rejects_impossible_transitions`               |
| Digest-mismatch fails closed                   | PASS   | `test_runtime_manifest_digest_mismatch_fails_closed`              |
| Bounded settings enforced (NaN, inf, ranges)   | PASS   | `test_settings_reject_out_of_range_and_nan`                       |
| End-to-end token streaming                     | PASS   | `test_runtime_generates_bounded_tokens_end_to_end`                |
| Session isolation across two sessions          | PASS   | `test_runtime_session_isolation_two_sessions_have_separate_histories` |
| K=16 + ACIS OFF certified inside runtime       | PASS   | `test_desktop_runtime_asserts_K16_and_ACIS_OFF`                   |
| Cancellation mid-generation                    | PASS   | `test_cancellation_mid_generation_returns_to_ready_and_preserves_committed` |
| Cancel before first token                      | PASS   | `test_cancel_before_first_token_still_returns_to_ready`           |
| Cancel wrong request returns False             | PASS   | `test_cancel_wrong_request_id_returns_false`                      |
| Shutdown during generation                     | PASS   | `test_shutdown_during_generation_completes_without_hanging`       |
| Recovery after failed load                     | PASS   | `test_recovery_after_failed_load_can_restart`                     |
| Chat UI module imports headless-safely         | PASS   | `test_chat_ui_module_imports_without_creating_window`             |
| New Session / Clear Conversation               | PASS   | `test_chat_controller_new_session_replaces_session_id`, `test_chat_controller_clear_conversation_resets_history` |
| 25 sequential requests + bounded RSS            | PASS   | `test_soak_25_sequential_generations_no_memory_growth_or_orphans` |
| 5 New Session cycles isolated                  | PASS   | `test_five_new_session_cycles_isolated`                           |
| Static network-token scan of `aeon.desktop.*`  | PASS   | `test_desktop_modules_have_no_outbound_network_reference`         |
| Desktop hot path never imports research modules| PASS   | `test_desktop_hot_path_does_not_import_research_only_modules`     |
| Release label does not misrepresent scale      | PASS   | `test_manifest_scale_labels_prevent_350M_misrepresentation`       |
| PyInstaller spec includes desktop modules      | PASS   | `test_pyinstaller_spec_includes_aeon_desktop_modules`             |
| PyInstaller spec bundles release-assets        | PASS   | `test_pyinstaller_spec_bundles_release_assets`                    |
| `--chat` mode + `--release-root` override      | PASS   | `test_entry_dispatches_chat_mode`, `test_entry_release_root_override` |
| Bundle excludes training artifacts             | PASS   | `test_release_bundle_excludes_forbidden_training_artifacts`       |

**Regression:** 59 test files, **673 explicit checks**, 0 failing.
Baseline at `377914b` was 627 checks; this campaign adds 46 desktop
checks and does not remove any prior tests.

---

## 2. Genuine Windows blocker

The following steps require a Windows runner and cannot be executed
from this Linux container:

| Step                                                                       | Blocker                                          |
| -------------------------------------------------------------------------- | ------------------------------------------------ |
| `pyinstaller --clean packaging/windows/Aeon.spec`                          | PyInstaller must run on Windows for a Windows Aeon.exe |
| `packaging/windows/build_installer.ps1` → Inno Setup                       | Inno Setup is Windows-only                       |
| Install `AeonSetup.exe`, launch `Aeon.exe --chat`, generate, close, reopen | Requires clean Windows machine                   |
| Uninstall + verify                                                          | Requires Windows                                 |

These are already the `windows-release.yml` (Tier A) + `windows-certification.yml`
(Tier B) workflows' responsibility. The desktop-tranche changes to
`packaging/windows/Aeon.spec` mean the next Tier A run will produce
an installer that includes the `aeon.desktop.*` modules + the
`release-assets/aeon-desktop-p2-proxy/` bundle.

### Exact next actions
1. On a Windows runner: `python scripts/export_aeon_desktop_model.py` — rebuild the release bundle deterministically.
2. `powershell.exe -File packaging/windows/build.ps1` — produce `dist/Aeon/Aeon.exe`.
3. `powershell.exe -File packaging/windows/build_installer.ps1` — produce `AeonSetup.exe`.
4. Install, launch `Aeon.exe --chat`, drive the vertical slice per §35.
5. Publish evidence to `docs/desktop/DESKTOP_ACCEPTANCE_REPORT.md`.

### Minimal reproduction — proving the Windows blocker is genuine
Attempting `pyinstaller --clean packaging/windows/Aeon.spec` from a
Linux container refuses because the spec pins the Windows-only
`_pyinstaller_hooks_contrib.stdhooks.hook-torch` torch bundling
which resolves torch's `.pyd` DLLs from the Windows torch wheel.
That is not a code defect — it is the correct spec for its target OS.

An **unsigned** installer is expected; §33 explicitly authorizes it
for the research preview and does not permit forged code-signing
claims.

---

## 3. Release identity

* **Release label:** `Aeon Desktop — Research Preview (7M P2 Proxy)`
* **Release channel:** `research_preview`
* **Tested scale:** `7M proxy`
* **Parameter count:** 7,015,366
* **Fixed K:** 16
* **ACIS default:** `OFF`
* **Network policy:** `offline_only`
* **Training code included:** `false`
* **Corpus included:** `false`
* **Sealed test included:** `false`
* **Optimizer state included:** `false`
* **Model export sha256:** stamped into
  `docs/desktop/desktop_release_evidence.json` at commit time.

Not the 350M primary model. Not Level 3 hidden-state proof.

---

## 4. Non-negotiables verified

* Every generated token runs through `aeon.hybrid.HybridModel`.
* K = 16 fixed in the config, in the checkpoint, in the runtime, in
  the manifest. Impossible to change from any settings surface.
* One shared broadcast per K-boundary — enforced by
  `HybridModel.forward` shape (nothing in this campaign touched
  `aeon.hybrid`).
* Recursion state fp32 — proven by parameter-dtype scan in the
  runtime test.
* Substrate autonomous — unchanged.
* Six V0.02.02 patches intact — `tests/test_six_patches.py` unmodified.
* ACIS OFF during every desktop-runtime generation — enforced by
  passing `shuttle=None` in `AeonDesktopRuntime._generate`.
* No `aeon.bypass.*` reachable from the desktop hot path — verified
  by static import scan.
* No outbound network reference in `aeon.desktop.*` — verified by
  static token scan.

---

## 5. What this release is NOT

* NOT the 350M primary model.
* NOT a Level 3 hidden-state proof.
* NOT a general-purpose instruction-following assistant. The 7M P2
  checkpoint was trained on six whole-work Project Gutenberg books at
  1,048,576 useful tokens. It produces plausible-english next-token
  continuations, not chat responses.
* NOT signed. Windows SmartScreen warnings are expected on install.

---

## 6. Compatibility contract for the later 350M swap

Documented in `docs/desktop/desktop_release_evidence.json` under
`compatibility_contract_for_later_350M`. Key points:

* Model class: `aeon.hybrid.HybridModel` (or subclass with same forward signature).
* Tokenizer: `aeon.tokenizer.AeonTokenizer` (SentencePiece).
* State-dict schema: flat `{str: torch.Tensor}` loadable via
  `torch.load(weights_only=True)` + `HybridModel.load_state_dict(strict=True)`.
* Manifest must declare `fixed_k=16`, `parameter_count` matches,
  `ACIS_default="OFF"`.
* Model swap is release-bundle-only. The desktop shell, chat runtime,
  IPC, and UI code do not change when the model changes.

That is the contract the 350M path needs to honor. It does not need
to touch any file under `aeon/desktop/`.
