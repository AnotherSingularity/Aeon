# Desktop Release Candidate — Aeon Desktop 7M Research Preview
### Reconciled (R0–R5) + Windows-State-B

**Ladder position:** **`FUNCTIONAL_UNPACKAGED_DESKTOP`** — the R5
1-hour soak completed cleanly with all 6 acceptance gates GREEN.
**`FUNCTIONAL_RELEASE_CANDIDATE`** requires WINDOWS-1..5 which
are STATE B (require a Windows runner).

**Reconciliation commit ledger** (added to `d91a836`):
```
DESKTOP-R0  — evidence audit → downgrade to FUNCTIONAL_LOCAL_BUILD
DESKTOP-R1  — authoritative desktop path = Architecture B (Tk)
DESKTOP-R2  — forward-logit equivalence + architectural trace (PROVEN)
DESKTOP-R3  — in-process supervision semantics (PROVEN)
DESKTOP-R4  — dynamic offline (PROVEN — socket + urllib patched)
DESKTOP-R5  — static readiness + 3600 s soak (see soak evidence)
WINDOWS-0..5 — STATE B (Windows execution required)
```

---

## 1. Verified end-to-end (reconciliation)

| Requirement                                                         | Verdict            | Test                                                                                            |
| ------------------------------------------------------------------- | ------------------ | ----------------------------------------------------------------------------------------------- |
| Authoritative desktop path is unambiguous                            | **YES** (Arch. B)  | `docs/desktop/DESKTOP_AUTHORITATIVE_PATH.md`                                                     |
| Inference export produces matching **logits**                        | **PROVEN**         | `test_R2_forward_logits_are_byte_identical`                                                     |
| Inference export produces matching **loss**                          | **PROVEN**         | `test_R2_forward_loss_matches`                                                                  |
| Deterministic 16-token greedy generation matches                     | **PROVEN**         | `test_R2_deterministic_generation_matches`                                                      |
| Transformer stream executes per generation                            | **PROVEN**         | `test_R2_architecture_trace_records_all_required_invariants`                                    |
| Substrate stream executes per generation                              | **PROVEN**         | same                                                                                            |
| Fixed K=16 boundary schedule executed                                 | **PROVEN**         | same (3 boundaries for seq_len=40)                                                              |
| Exactly one shared broadcast per boundary                             | **PROVEN**         | same                                                                                            |
| ACIS remains OFF throughout desktop generation                        | **PROVEN**         | `test_R2_desktop_runtime_generation_is_ACIS_OFF`                                                |
| UI streams authentic output                                           | **PROVEN**         | `test_runtime_generates_bounded_tokens_end_to_end` + chat_ui `_drain_events`                    |
| Stop cancels between token steps                                     | **PROVEN**         | `test_cancellation_mid_generation_returns_to_ready_and_preserves_committed`                     |
| Sessions isolate                                                      | **PROVEN**         | `test_runtime_session_isolation_two_sessions_have_separate_histories`                           |
| New Session / Clear / Reset                                           | **PROVEN**         | `test_chat_controller_new_session_replaces_session_id`, `..._clear_conversation_resets_history` |
| Runtime restart from failed load                                     | **PROVEN**         | `test_recovery_after_failed_load_can_restart`                                                    |
| Duplicate request rejected while active                              | **PROVEN**         | `test_R3_duplicate_request_rejected_while_one_active`                                            |
| Release path traversal rejected                                       | **PROVEN**         | `test_R3_release_path_traversal_rejected`                                                        |
| No arbitrary path / eval / exec surface                              | **PROVEN**         | `test_R3_no_arbitrary_path_argument_on_public_api`                                              |
| Bounded event queue never blocks runtime                             | **PROVEN**         | `test_R3_bounded_queue_never_blocks_runtime_on_overflow`                                        |
| **Dynamic** network denial (socket + urllib patched)                 | **PROVEN**         | `test_R4_desktop_pipeline_runs_without_any_outbound_network_attempt`                            |
| No thread leak across 5 gen+cancel cycles                            | **PROVEN**         | `test_R4_no_new_thread_leaks_after_repeated_gen_cancel`                                          |
| Continuous 1-hour soak (3600.58 s)                                    | **PROVEN** — 2,712 gens / 771 cancels / 915 resets / 902 new sessions / RSS plateau at 679 MB after 10-min warm-up / 1 thread constant / 0 errors / all 6 acceptance gates GREEN. See `docs/desktop/DESKTOP_FULL_SOAK_REPORT.md` |
| Runtime CRASH recovery (kill mid-generation, shell dies)             | **DISCLOSED**      | in-process design trade-off; documented in AUTHORITATIVE_PATH §5                                 |
| Orphan-process detection                                             | **NOT_APPLICABLE** | in-process design (no subprocess)                                                                |

## 2. State B (Windows) — remaining gates

Not verifiable from this Linux container. Documented with exact
next commands in `docs/desktop/DESKTOP_WINDOWS_BUILD_REPORT.md`:

| Gate                                                                  | Status  |
| --------------------------------------------------------------------- | ------- |
| Frozen runtime builds via `packaging/windows/build.ps1`                | NOT_RUN |
| Frozen `Aeon.exe --chat` runs OUTSIDE the repo checkout                | NOT_RUN |
| Inno installer builds via `packaging/windows/build_installer.ps1`      | NOT_RUN |
| Clean-Windows-machine install + vertical slice per §35                 | NOT_RUN |
| Ten application restart cycles                                          | NOT_RUN |
| Injected runtime-crash trials                                          | NOT_RUN |
| Invalid-release startup trials (digest mismatch on frozen build)       | NOT_RUN |
| Upgrade path (old → new installer)                                     | NOT_RUN |
| Uninstall clean removal                                                | NOT_RUN |

## 3. Release identity — unchanged

* Release label: `Aeon Desktop — Research Preview (7M P2 Proxy)`
* Tested scale: `7M proxy` — **NOT** the 350M primary model.
* Parameter count: 7,015,366 · Fixed K: 16 · ACIS default: **OFF**.
* Network policy: `offline_only`.
* Model export: `sha256:c10350ac5569cd44e93226b40b1aa4cd0b8b2773ebe45401719946038015f1e4`
  (from an in-container export; see the byte-hash non-determinism note
  in `docs/desktop/desktop_windows_evidence.json` — consumers verify
  against the shipped manifest, not against a repo-pinned hash).
* Tokenizer: `sha256:064ab6a98ee4b8177249f14dc09e60ccaa9986b66cb84a4072c68dc4de533481`.
* Source P2 checkpoint: `sha256:962fcd5e65a88e3b6d061e73968bea7eb3581c4d8a15b49be82427004db9fc3c`.

## 4. Research-claim status — unchanged

* Level 2 OBSERVATIONAL_EVIDENCE. Level 3 status = CANDIDATE_NOT_CLOSED.
* Narrowed causal statement (from `docs/latent_bypass/claim_level.json`):
  the learned shared Recursion broadcast is causally necessary for the
  achieved held-out performance of the fixed 7M P2 checkpoint, and a
  same-norm irrelevant replacement does not preserve that performance.
* Desktop reports carry only that narrowed statement. No hidden-state
  bypass has been proven.

## 5. Six V0.02.02 patches + K=16 + ACIS OFF + sealed test

All non-negotiables preserved across the reconciliation:

* K = 16 (config + checkpoint + runtime + manifest + R2 trace)
* One broadcast per K-boundary (R2 trace)
* Recursion state fp32 (R2 trace)
* Substrate autonomous — unchanged; `aeon/substrate/` not modified
* ACIS OFF during desktop generation — R2 wrap on `HybridModel.forward`
  verified `shuttle=None` on every call
* Six V0.02.02 patches — `tests/test_six_patches.py` unchanged, passes
* Sealed TEST partition — never opened by any desktop code

## 6. Regression

Baseline at `d91a836`: 673 checks / 0 failing.
Reconciliation adds 8+9+7+8 = 32 checks with no test removed.
Expected final: **705+ checks** on the reconciliation head
(exact final total confirmed by the closing regression run and
recorded in `docs/desktop/desktop_release_evidence.json`).

## 7. Exact next actions

**On a Windows runner:**
1. `git switch claude/aeon-desktop-7m-validation && git pull --ff-only`
2. `python scripts/export_aeon_desktop_model.py` — regenerate bundle
3. `powershell.exe -File packaging/windows/build.ps1` — frozen runtime
4. `powershell.exe -File packaging/windows/build_installer.ps1` — installer
5. Clean-Windows install; run vertical slice per §35.
6. Populate `docs/desktop/DESKTOP_WINDOWS_ACCEPTANCE_REPORT.md` with
   evidence.
7. Set `docs/desktop/desktop_status.json.current_status =
   FUNCTIONAL_RELEASE_CANDIDATE`.

## 8. 350M model-swap readiness

* Compatibility contract locked in
  `docs/desktop/desktop_release_evidence.json.compatibility_contract_for_later_350M`.
* Migration is release-bundle-only when the model is a HybridModel
  subclass with the same forward signature.
* Migration REQUIRES the runtime move to a supervised subprocess
  (currently in-process). The `aeon.launcher.gui` + `aeon.job.worker`
  supervisor pattern is the reference; `AeonDesktopRuntime` can be
  externalized without touching `aeon.desktop.chat_ui` or the event
  schema.
