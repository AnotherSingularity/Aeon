# V0.02.03 Upgrade — Definition of Done

Directive §19 enumerates 26 conditions. Each row below identifies the evidence.

| # | condition | evidence | status |
|---|---|---|---|
| 1 | The two independent streams remain intact | `tests/test_stream_independence.py` (5) + `docs/TOPOLOGY_MAP.md` | **DONE** |
| 2 | Both streams feed Recursion through authorized paths | `tests/test_recursion_topology.py::test_single_broadcast_shared_source` + AST assertions | **DONE** |
| 3 | Recursion retains its existing single broadcast | `tests/test_recursion_topology.py::test_single_broadcast_shared_source` — no dual-head modules | **DONE** |
| 4 | Both streams consume that single broadcast | `hybrid.py:132` (`cond_proj(h_cond)`) + `hybrid.py:139` (`inject_cols.append(h_cond)`); asserted by the same AST test | **DONE** |
| 5 | `K=16` is preserved and tested | `aeon/hybrid.py:68`, `configs/aeon_350m_primary.yaml:19`, `configs/aeon_v1.yaml`, `configs/aeon_smoke_e5.yaml`; `tests/test_recursion_topology.py::test_K_is_16_and_not_adaptive`; `tests/test_config_invariants.py::test_every_config_pins_K_to_16` | **DONE** |
| 6 | Recursion state remains `fp32` | `scripts/train.py:116`, `scripts/infer.py:43`, `scripts/diagnose.py`; `tests/test_recursion_topology.py::test_recursion_stays_fp32_after_cast` | **DONE** |
| 7 | The contractive certificate passes and fails closed | `tests/test_recursion_topology.py::test_certificate_holds_by_construction` + `test_certificate_fails_closed_on_forced_violation` | **DONE** |
| 8 | The substrate remains autonomous | `tests/test_stream_independence.py::test_substrate_feedback_uses_no_transformer_names` + `test_matrix_cell_step_signature_is_substrate_only` + `test_substrate_readout_invariant_to_transformer_within_window` | **DONE** |
| 9 | Substrate state follows parameter dtype | `tests/test_six_patches.py::test_4e_substrate_state_dtype` | **DONE** |
| 10 | All six V0.02.02 patches identified and regression-tested | `tests/test_six_patches.py` (6 named tests, one per patch) + `docs/PRESERVATION_MANIFEST.md` | **DONE** |
| 11 | Permanent observability overhead is below 15 % | `tests/test_observability.py::test_permanent_instrumentation_overhead_under_15_percent` (assertion at worst-case sample_every=1) | **DONE** |
| 12 | Instrumentation does not alter model semantics or gradients | `tests/test_observability.py::test_instrumentation_on_off_bitexact_outputs_and_grads` | **DONE** |
| 13 | Heavy diagnostics are disabled by default | `scripts/diagnose.py` runs offline on saved checkpoints; `scripts/train.py` uses only permanent instrumentation (no every-layer hooks) — §9 escalation protocol documented in `docs/OPERATIONS.md` | **DONE** |
| 14 | Checkpoints are atomic and integrity-checked | `aeon/checkpoint.py::atomic_save` + `.sha256` sidecar + `.prev` retention; `tests/test_checkpoint.py::test_atomic_save_survives_interrupted_write` + `test_atomic_save_preserves_prior_on_new_save` | **DONE** |
| 15 | Resume equivalence passes within declared bounds | `tests/test_checkpoint.py::test_resume_equivalence_bounded` (pre-resume bit-equal; post-resume < 1e-4) | **DONE** |
| 16 | Local-security boundaries pass | `tests/test_checkpoint.py::test_strict_load_uses_weights_only_or_hardened`, `test_reject_incompatible_metadata_*`, `test_reject_corrupt_sha256`, `test_missing_sha256_rejected_when_required` + `docs/SECURITY_MODEL.md` | **DONE** |
| 17 | Offline diagnostics work on saved checkpoints | `scripts/diagnose.py`; `tests/test_diagnose.py::test_diagnose_all_does_not_mutate_checkpoint` + `test_diagnose_interventions_are_evaluation_only` | **DONE** |
| 18 | A bounded representative training run succeeds | `docs/E5_CERTIFICATION.md` + `docs/e5_evidence.json` — verdict PASS | **DONE** |
| 19 | The training run crosses multiple Recursion boundaries | E5: 20 steps × 4 windows/step = 80 boundaries | **DONE** |
| 20 | Validation, checkpoint, restoration, and continued training all succeed | E5 scenarios 4, 6, 7, 8, 9 all PASS | **DONE** |
| 21 | The primary training configuration is complete | `configs/aeon_350m_primary.yaml` versioned; `docs/e6_parameter_accounting.json` records 350.28 M | **DONE** |
| 22 | Fresh-start and resume commands are proven | Documented in `docs/OPERATIONS.md`; verified in E5 scenarios 1-7 and the E3 end-to-end train+resume smoke | **DONE** |
| 23 | Documentation matches the implementation | Every doc references code by file and line; every code addition has a documented invariant target in `docs/PRESERVATION_MANIFEST.md` | **DONE** |
| 24 | Efficiency claims remain within the evidence | `docs/OPERATIONS.md` §5 + `docs/COMMIT_REPORT.md` known limitations + `docs/PROXY_CAMPAIGN_PLAN.md` claim-boundary section | **DONE** |
| 25 | The full automated suite passes | 61 / 61 pass; recorded in `docs/e7_final_evidence.json` | **DONE** |
| 26 | No unresolved architectural, numerical, security, checkpoint, or runtime blocker remains | E0 recorded no pre-existing blocker; E1-E5 introduced none; E5 verdict PASS on all ten scenarios; final suite 61/61 | **DONE** |

## Overall verdict

**The upgrade satisfies the complete definition of done.** All 26 conditions
are met with named evidence. No preservation invariant was traded for speed
or convenience. No hidden dual-head, no adaptive clock, no bypassed
certificate, no unpinned tokenizer, no leaked cross-stream read.

The repository is ready for the primary Aeon training campaign per
`docs/OPERATIONS.md`.
