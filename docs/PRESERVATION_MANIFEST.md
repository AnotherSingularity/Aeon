# Aeon V0.02.03 — Preservation Manifest

**Version:** 1 (E0). This file is machine-readable via `docs/preservation.json`.

Every invariant below is a hard preservation rule for the E0–E7 upgrade. Each row identifies the invariant, its code location, its existing regression test (or the E1 target if a new test is needed), the failure mode it prevents, and its runtime scope.

## Legend

- **scope** — I(init) F(forward) B(backward) R(state reset) S(save) L(load) N(inference) D(dtype transition)

## Invariants

| id | invariant | code location | test (existing / E1) | failure prevented | scope |
|---|---|---|---|---|---|
| **P-4a** | γ recast to fp32 after `model.to(dtype)`; also mirrored in `infer.py` for parity | `scripts/train.py:119`, `scripts/infer.py` | `test_aeon_sanity::test_gamma_updates_bf16_trap` (existing) + `test_six_patches::test_4a_gamma_recast_after_cast` (E1 new) | bf16 γ ULP > AdamW step → γ freezes at 1/32 | D |
| **P-4b** | γ Parameter created `dtype=torch.float32` | `aeon/transformer.py:246` | `test_six_patches::test_4b_gamma_param_dtype` (E1 new) | reproducibility that γ intent is fp32 in code (belt-and-suspenders with P-4a) | I |
| **P-4c** | `inject()` residual add is fp32: `(hidden.float() + γ · write_proj(signal).float()).to(dtype)` | `aeon/transformer.py:266` | `test_six_patches::test_4c_inject_fp32_add` (E1 new) | γ's gradient path degrades in bf16 → gradient underflow | F |
| **P-4d** | `write_proj` random init `normal_(std=0.02)`, NOT zeros | `aeon/transformer.py:241` | `test_aeon_sanity::test_gradient_flows_everywhere` + `test_six_patches::test_4d_write_proj_random_init` (E1 new) | γ×write_proj mutual-zero deadlock → recurrent branch gets no gradient | I |
| **P-4e** | Substrate `reset()` uses parameter dtype (both cells) | `aeon/substrate/matrix_cell.py:98`, `vector_cell.py:53` | `test_six_patches::test_4e_substrate_state_dtype` (E1 new) | fp32 state × bf16 params → mixed-dtype matmul crash on first forward | R |
| **P-4f** | rotary `inv_freq` fresh fp32 inside forward, no `register_buffer` | `aeon/transformer.py:94`; ZERO `register_buffer` calls in the file (verified) | `test_six_patches::test_4f_rotary_inv_freq_fresh_fp32` (E1 new) | `model.to(bf16)` downcasts a buffered inv_freq → wrong rotary → cascading error | F |
| **P-K16** | slow clock `K=16` (default in code, set in configs, not adaptive) | `aeon/hybrid.py:68`, `configs/aeon_350m.yaml:14`, `configs/aeon_v1.yaml:6` | `test_recursion_topology::test_K_is_16_and_not_adaptive` (E1 new) | drift from certified slow-clock cadence → cert & training pattern change | F I |
| **P-fp32-rec** | Recursion module stays fp32 after global dtype cast | `scripts/train.py:116`, `scripts/infer.py:43` | `test_recursion_topology::test_recursion_stays_fp32_after_cast` (E1 new) | bf16 Cayley solve / SVD unsupported or unstable → certificate breaks | D |
| **P-cert** | contractive certificate: `σ(W_h) < margin_h`, `σ(W_c) < margin_c` — structural (`sigmoid·MARGIN·Cayley·diag(tanh)`), audited by `RecursionJoiner.audit`; fail-closed downstream | `aeon/recursion.py:135-141, 173-…` | `test_aeon_sanity::test_certificate_holds_at_init` + `test_feedback::test_certificate_holds_all_modes` (existing) + `test_recursion_topology::test_certificate_fails_closed` (E1 new) | silent drift or approximation past σ<margin → contractive guarantee lost | F N D |
| **P-single-bcast** | one Recursion broadcast `h_{w-1}` consumed by both streams (`cond_proj` + `inject`) — no `J_S`/`J_T` split | `aeon/hybrid.py:131-139, 154` | `test_recursion_topology::test_single_broadcast_shared_source` (E1 new) | dual-head silently introduced → topology changes without notice | F |
| **P-parallel** | substrate & transformer are independent forward paths; no direct cross-stream tensor reads outside Recursion | `aeon/hybrid.py::HybridModel.forward` (whole function), no import cycle between `aeon/substrate/*` and `aeon/transformer.py` at forward-time | `test_stream_independence::test_no_direct_cross_stream_read` (E1 new; combines AST + import-graph check) | any cross-stream link that bypasses Recursion → violates §3.1 | F |
| **P-sub-autonomy** | substrate & its gate see substrate-internal signals + the authorized Recursion broadcast only — no transformer state | `aeon/substrate/feedback.py::forward` (`base` = own readout), `aeon/hybrid.py:131-139` (`cond_in` = broadcast, not transformer) | `test_stream_independence::test_substrate_autonomy` (E1 new) | substrate reads transformer entropy / logits / hidden → §3.6 violated | F |
| **P-ckpt** | resumable checkpoint preserves all state required by directive §10.1 | `scripts/train.py::save_checkpoint`/`main` (pre-E3) → `aeon/checkpoint.py::atomic_save/load` (E3) | `test_checkpoint::test_atomic_save_and_resume_equivalence` (E3 new) | partial resume → training divergence, silent data-position reset, wrong config | S L |
| **P-security** | training/inference load with `weights_only=True`, reject incompatible metadata, no network / shell / arbitrary FS access | `aeon/checkpoint.py::load` (E3 new) | `test_checkpoint::test_reject_incompatible_metadata`, `test_checkpoint::test_load_uses_weights_only` (E3 new) | pickle-code execution from a hostile checkpoint, silent tokenizer/vocab swap | L |

## Sign-off (E0)

At E0 all 14 invariants are traceable to code (right column verified in the audit). The E1 checkpoint adds the "(E1 new)" tests. E3 adds the checkpoint + security tests. No architectural code is touched in E0.

## Machine-readable index

See `docs/preservation.json` — one object per row above with `id`, `location`, `test`, `failure`, `scope`.
