# Post-Campaign Claim Reconciliation

**Reconciliation source commit:** `9b9ec80`
**Prior claim at commit `739daf6`:** Level 3 CAUSAL_CHECKPOINT_EVIDENCE
**Reconciled claim:** **Level 2 OBSERVATIONAL_EVIDENCE ·
                        Level 3 status = CANDIDATE_NOT_CLOSED**

This document walks the L5 evidence at `9b9ec80` against the
preregistered Level 3 requirements written into the L3 calibration
lock, and records why the earlier Level 3 report was premature.

Rules followed (from the reconciliation directive):

* §1 — reconcile claim files against preregistered Level 3 requirements.
* §3 — do **not** rerun training and do **not** alter sealed TEST results.
* §3 — do **not** change any threshold, barrier, intervention, or
  sample selection.
* §4 — if any required Level 3 item is absent, downgrade to Level 2 +
  record missing gates explicitly.
* §5 — narrow the causal statement.
* §7 — record SHUFFLE_BROADCAST as unresolved.
* §8 — record the tested scale as the 7M proxy, not the 350M primary.
* §9 — preserve ACIS OFF as default.

---

## 1. The three preregistered Level 3 requirements

From `docs/latent_bypass/L3_CALIBRATION_LOCK.json.statistical_plan` and
the parent directive §18 (locked before TEST access):

| # | Requirement                                                                                                                     |
| - | ------------------------------------------------------------------------------------------------------------------------------- |
| a | Paired barrier-region vs nonbarrier-region intervention effects, using the L2 barrier registry calibrated on CALIBRATION.       |
| b | A locked null region for ΔL_c. The lock records `null_region = [-0.01, +0.01]` in nat units.                                    |
| c | A confidence bound produced by the declared method. The lock records `confidence_method = "paired-batch resampling; report mean + range"`. |

---

## 2. Audit of the persisted L5 evidence

Inspecting `docs/latent_bypass/l5_causal_evidence.json` at commit
`9b9ec80` shows exactly which of the three gates are backed by raw
evidence:

### 2.a — Barrier-region stratification: **ABSENT**

* The L5 script emitted per-intervention aggregate ΔL_c across all 24
  TEST batches. Every batch was pooled; no barrier label was applied.
* The L2 barrier registry was NOT calibrated on CALIBRATION for this
  campaign. `benchmarks/latent_bypass/barriers.json` still carries
  `threshold_value: null` for every barrier. The registry's digest
  is recorded in the calibration lock, but the calibration step
  itself never ran.
* Reconciliation §3 forbids calibrating barriers post-hoc.

### 2.b — Locked null region: **PRESENT (partial)**

* `[-0.01, +0.01]` in nat units is recorded in the lock.
* The report at commit `739daf6` interpreted "|ΔL_c| > 0.01" as the
  positive gate, which is a correct application of the locked null
  region at the pooled (all-batches) level.
* **However**, this was never applied per-region — because §2.a's
  regions do not exist.

### 2.c — Paired-batch resampling CI: **NOT COMPUTABLE from persisted evidence**

* `results.NONE.per_batch_loss` contains all 24 baseline batch losses.
* `results.<intervention>` contains only `mean_loss`, `per_batch_loss_min`,
  `per_batch_loss_median`, `per_batch_loss_max`, and `delta_L_c_vs_none`.
* Per-batch losses for the interventions were **not** persisted.
* Without per-batch intervention losses aligned to per-batch NONE
  losses, paired ΔL_c per batch is not reconstructable. Paired-batch
  resampling as declared cannot be executed on the committed evidence.
* Reconciliation §3 forbids re-running L5 to backfill.

---

## 3. Verdict

Under reconciliation §4:

    achieved_claim_level = 2  (OBSERVATIONAL_EVIDENCE)
    level_3_status       = CANDIDATE_NOT_CLOSED
    level_3_missing_gates = [ barrier-region stratification,
                              paired-batch resampling CI ]

The evidence that survives at Level 2:

* Two large aggregate causal signals on TEST, above the locked null
  region, from the fixed P2 checkpoint:
  - `ZERO_BROADCAST`                  ΔL_c = +0.6910 nat
  - `NORM_MATCHED_IRRELEVANT_STATE`   ΔL_c = +0.7102 nat
* The two signals are close in magnitude, which is the intended
  norm-controlled reading.

What does NOT survive at Level 3:

* Barrier-selective causal signal (never computed).
* A resampled confidence bound (not computable post-hoc).

---

## 4. Narrowed causal statement (per §5)

Replace the earlier broad "the broadcast content matters" phrasing with:

> **The learned shared Recursion broadcast is causally necessary for
> the achieved held-out performance of the fixed 7M P2 checkpoint,
> and a same-norm irrelevant replacement (`NORM_MATCHED_IRRELEVANT_STATE`)
> does not preserve that performance.**

Per §6, this is **not** a hidden-state-bypass claim. That claim
requires the full L3 gate set (barrier selectivity + resampled CI +
cross-seed replication + net-efficiency).

---

## 5. SHUFFLE_BROADCAST is unresolved (per §7)

The measured aggregate effect was ΔL_c = +0.0001 nat — inside the
null region. This is recorded, but its interpretation is unresolved.
Required controls before drawing a scientific conclusion:

* semantic-distance shuffle (permute across the entire TEST corpus,
  not within a single batch);
* temporal-shuffle (permute across boundaries within a batch, not
  across batch elements);
* covariance-matched shuffle (draw permutations that preserve the
  broadcast's empirical covariance);
* cross-seed replication (single seed cannot rule out a lucky
  permutation).

None of these controls have been executed. Do not treat
"SHUFFLE ≈ 0" as evidence that batch-position alignment is irrelevant.

---

## 6. Tested scale (per §8)

* Model: **7M-parameter proxy configuration**
  (`configs/latent_bypass/aeon_lbc1_proxy.yaml`, hidden 256, 4 layers,
  h_rec 128).
* This is **not** the 350M primary model. Every quantitative
  conclusion here is scoped to the 7M proxy. Any generalization to
  the primary model requires re-running the campaign under the
  primary configuration.

---

## 7. ACIS status preserved (per §9)

Unchanged from ACIS-8 and from the workload certification at commit
`8a16dc8`:

* **Default: OFF.**
* OBSERVE — overhead target failed (median 7.40 % > 3 %, p95 4.22 %
  under 5 % ceiling). Semantic equivalence to OFF confirmed.
* BUCKET — no demonstrated benefit (median overhead 5.66 %).
  Semantic equivalence to OFF confirmed.
* CONVEYOR — refused.

---

## 8. Exact remaining Level 3 gates

To close Level 3 without touching the sealed TEST partition twice for
the same experiment version, the next campaign must (at a minimum):

1. Calibrate the L2 barrier registry on CALIBRATION per its own
   calibration procedure. Commit the calibrated registry (and updated
   registry digest) before opening TEST.
2. Rewrite the L5 runner to persist per-batch per-intervention losses
   so paired ΔL_c per batch can be aligned to the paired NONE loss.
3. Execute the declared confidence method — paired-batch
   resampling — and report mean + resampled range against the locked
   null region.
4. Stratify every ΔL_c and every resampled CI by barrier vs
   non-barrier region.
5. Preregister these steps as a new experiment version in a fresh
   L3 calibration lock. The current lock's fields cannot be silently
   changed; a lock revision is a new experiment identity.

Once (1)–(4) land and (5) is committed, Level 3 CANDIDATE_NOT_CLOSED
becomes eligible for re-audit against the same sealed TEST — or, if
the sealed TEST is considered exhausted by this campaign, a fresh
sealed partition must be introduced.

---

## 9. What this reconciliation does not touch

* No training was rerun.
* No sealed TEST results were altered.
* No threshold, barrier, intervention, or sample selection was
  changed.
* No `aeon/` source was modified.
* No test file was added or removed.
* K = 16 remains fixed. Six V0.02.02 patches remain intact.
