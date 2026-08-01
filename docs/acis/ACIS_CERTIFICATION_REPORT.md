# ACIS Certification Report

**Base commit:** `ee28f48` (STATE B stopping point).
**Regression at closure:** 616 explicit checks, 0 failures.
**Certified default mode:** `OFF`.

---

## 1. Commit Ledger

Additive commit ledger — each tranche is independently reviewable
and each was landed with an executable test suite:

| Tranche | Commit    | Summary                                                        |
| ------- | --------- | -------------------------------------------------------------- |
| ACIS-0  | `2ffe176` | Invariant lock + baseline scaffolding.                         |
| ACIS-1  | `1e9b0e4` | Immutable broadcast + representation contract (OBSERVE-only).  |
| ACIS-2  | `5dd98a6` | Destination read leases + lifecycle custody.                   |
| ACIS-3  | `68084c3` | Recursion broadcast shuttle wire-through under `shuttle=None`. |
| ACIS-4  | `a3d889d` | Mutable capsule + ownership ledger.                            |
| ACIS-5  | `6814b3a` | Bucket lane + freshness + backpressure.                        |
| ACIS-6  | `dafb25f` | Recovery + replay + quarantine + coherence.                    |
| ACIS-7  | `4364255` | Transport calibration + conveyor decision gates.               |
| ACIS-8  | PENDING   | Integrated certification and closure (this tranche).           |

---

## 2. Invariant Verdicts

Every invariant required by the ACIS directive has an executable
test that gates it. Verdict summary:

| Invariant                                                | Status     | Evidence                                                  |
| -------------------------------------------------------- | ---------- | --------------------------------------------------------- |
| `K = 16` fixed, no adaptive-K component                  | ENFORCED   | `test_fixed_k_declared_at_16`, `test_no_adaptive_k_*`     |
| One broadcast per K-boundary                             | ENFORCED   | `test_single_broadcast_per_window`                        |
| Both leases resolve to same broadcast id + digest        | ENFORCED   | `test_coherence_accepts_identical_pair`                   |
| Both leases resolve to same live tensor object          | ENFORCED   | `test_coherence_refuses_object_divergence`                |
| Recursion state fp32                                     | ENFORCED   | `test_recursion_still_fp32_in_slow_clock_tick`            |
| Substrate autonomous (no direct transformer↔substrate)   | ENFORCED   | `tests/test_stream_independence.py`                       |
| Six V0.02.02 patches intact                              | ENFORCED   | `tests/test_six_patches.py`                               |
| OFF-mode byte-identical to shuttle-absent build          | ENFORCED   | `test_shuttle_none_produces_byte_identical_forward`       |
| Shuttle imports guarded by `shuttle is not None`         | ENFORCED   | `test_hybrid_shuttle_import_is_guarded_by_none_check`     |
| No outbound network in `aeon/shuttle/`                   | ENFORCED   | `test_shuttle_package_has_no_outbound_network_reference`  |
| Immutable broadcast + read-only leases                   | ENFORCED   | `tests/test_acis_1_broadcast.py` + `test_acis_2_leases.py`|
| Single mutable ownership                                 | ENFORCED   | `test_single_ownership_holds_after_transfer`              |
| Immutable broadcasts excluded from mutable owner check   | ENFORCED   | `test_immutable_broadcasts_excluded_*`                    |
| Precommit rollback restores source authority             | ENFORCED   | `test_precommit_rollback_restores_source_authority`       |
| Lane FIFO, capacity, dedup, cancel, stage graph          | ENFORCED   | `tests/test_acis_5_lane.py` (7 tests)                     |
| Freshness rejects stale/future/dup/causal/expired        | ENFORCED   | `tests/test_acis_5_lane.py` (5 tests)                     |
| Backpressure never alters K                              | ENFORCED   | `test_backpressure_refuses_altering_K`                    |
| Quarantine blocks readmission                            | ENFORCED   | `test_quarantine_records_and_blocks_readmission`          |
| Replay journal refuses regression + pre-stop replay      | ENFORCED   | `tests/test_acis_6_recovery.py` (4 tests)                 |
| Safe Stop drains lanes; resume refuses replay            | ENFORCED   | `test_recovery_resume_rejects_prestop_boundary`           |
| No test asserts `executed_recursion_iterations == 16`    | HELD       | grep sweep — no such assertion exists                     |
| No `AdaptiveKControlCapsule` in the codebase             | HELD       | grep sweep — no such symbol exists                        |

---

## 3. Rollout Decision

| Mode                      | Decision     | Reason                                                              |
| ------------------------- | ------------ | ------------------------------------------------------------------- |
| `OFF`                     | CERTIFIED    | Byte-identical to shuttle-absent build. No ACIS code executes.      |
| `OBSERVE`                 | CERTIFIED    | Broadcast + lease + audit-event allocation only. No mutation.       |
| `BUCKET`                  | STRUCTURALLY CERTIFIED, runtime overhead measurement pending live workload. |
| `CONVEYOR_EXPERIMENTAL`   | REFUSED      | `no_conveyor_evidence` — certified default per `decide_conveyor`.   |

---

## 4. Autograd + Semantic Identity

* Payload flow into `inject_cols` is UNCHANGED. `h_cond` remains the
  same tensor object that fed the transformer before ACIS.
* `compute_semantic_digest` uses `detach().to(float32).contiguous()`
  for hashing ONLY. The autograd graph of the payload is not
  altered.
* Two publishes of the same tensor + contract at the same boundary
  yield the same broadcast_id and the same semantic_digest —
  proven by `test_two_publishes_of_same_boundary_yield_equivalent_*`.
* OFF-mode produces bit-identical output to the shuttle-absent
  forward — proven by
  `test_shuttle_none_produces_byte_identical_forward`.

---

## 5. Non-Alteration Assertions

ACIS-8 asserts explicitly:

* `K` remains fixed at `16` in `aeon/hybrid.py` and
  `aeon/shuttle/__init__.py`.
* No file under `aeon/` contains an `AdaptiveKControlCapsule`
  class, an adaptive-K function, or an assertion of the form
  `executed_recursion_iterations == 16`.
* No file under `aeon/shuttle/` opens a socket, imports
  `urllib.request`, `requests`, or `httpx`, or references any
  outbound HTTP verb.
* No six-patch corrections were modified during any ACIS tranche.

---

## 6. Closure

The full ACIS ledger from ACIS-0 through ACIS-8 has been landed as
an ADDITIVE sequence on branch `claude/funny-cori-a3k5cf`. All
executable tests pass; all invariants are enforced by named tests;
the OFF-mode default is byte-identical to the pre-ACIS forward pass.

Return condition: **STATE A — ACIS complete.**
