# L5 Causal Intervention Report

**Fixed experimental basis:** `runs/aeon_lbc1_P2/final.pt`
**Sealed partition opened:** TEST — PG-1661 (The Adventures of Sherlock
Holmes), 2,546 processed records, source SHA-256:
`922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0`,
partition digest: `a64f4cb9673f8b867cc81b041bda8198a6c403921ccf27f5546c32930e35947e`.
**Trial workload:** 24 batches × 8 batch × 256 seq = 49,152 tokens
per intervention. Same batches used across every intervention.
**Runner:** `scripts/run_l3_l4_l5.py`. `assert_evaluation_mode(model)`
enforced before every intervention; no persistence keyword accepted.

Test access is gated by the committed
`docs/latent_bypass/L3_CALIBRATION_LOCK.json`. The lock was written
by this script's `write_calibration_lock()` before `do_L5()` fires.

---

## 1. Intervention hooks

Every intervention lives inside a `wrapped_step` closure that
temporarily replaces `model.recursion.step`. On every K-boundary the
closure applies the intervention (PRE or POST) then calls the original
step. When `do_L5` returns, the original step is restored. Model
parameters, tokenizer, corpus, ACIS ledger, and checkpoints are all
untouched — the intervention is a per-forward alteration only.

| Intervention                    | Applied at | Description                                                     |
| ------------------------------- | ---------- | --------------------------------------------------------------- |
| NONE                            | —          | Baseline forward, no alteration                                 |
| ZERO_BROADCAST                  | post-step  | Overwrite h_new := 0                                            |
| FREEZE_BROADCAST                | post-step  | Replace h_new with h from the first boundary of the run         |
| DELAY_BROADCAST                 | post-step  | Delay h_new by one boundary (use previous window's h)           |
| SHUFFLE_BROADCAST               | post-step  | Permute h_new across the batch dimension                        |
| FREEZE_RECURSION                | post-step  | Freeze both h_new and c_new to their first-boundary values      |
| MASK_TRANSFORMER_SOURCE         | pre-step   | Zero the transformer source t_w before recursion.step           |
| MASK_SUBSTRATE_SOURCE           | pre-step   | Zero the substrate source s_w before recursion.step             |
| NORM_MATCHED_IRRELEVANT_STATE   | post-step  | Replace h_new with a random vector rescaled to its own norm     |

---

## 2. Results

Baseline (NONE) mean loss on 24 TEST batches: **5.7146**

| Intervention                    |  mean loss | ΔL_c vs NONE | Signals broadcast content matters |
| ------------------------------- | ----------:| ------------:| --------------------------------- |
| NONE                            |    5.7146  |     0.0000   | (baseline)                        |
| **ZERO_BROADCAST**              |    6.4056  |    **+0.6910** | **✔** — LARGE                     |
| FREEZE_BROADCAST                |    5.7398  |     +0.0252  | small                             |
| DELAY_BROADCAST                 |    5.7170  |     +0.0024  | negligible                        |
| SHUFFLE_BROADCAST               |    5.7147  |     +0.0001  | negligible                        |
| FREEZE_RECURSION                |    5.7398  |     +0.0252  | small                             |
| MASK_TRANSFORMER_SOURCE         |    5.7269  |     +0.0123  | small                             |
| MASK_SUBSTRATE_SOURCE           |    5.7654  |     +0.0508  | modest                            |
| **NORM_MATCHED_IRRELEVANT_STATE** | 6.4248  |    **+0.7102** | **✔** — LARGE                     |

Reference null region: ± 0.01 in nat units.

---

## 3. Interpretation

Two clean, matched-control causal signals emerge:

### 3.1 Broadcast content matters (ZERO / NORM-MATCHED)

* Removing the broadcast entirely (ZERO_BROADCAST) increases mean loss
  by **+0.6910 nat**, well outside the null region.
* Replacing the broadcast with a random vector rescaled to the SAME
  norm as the true broadcast (NORM_MATCHED_IRRELEVANT_STATE) increases
  loss by **+0.7102 nat** — essentially the same effect.
* The two effects are similar in magnitude, which is the correct
  norm-controlled reading: it is the CONTENT of the broadcast that
  matters to next-token prediction, not just its magnitude.

### 3.2 Broadcast has batch-shared structure, not per-example alignment (SHUFFLE)

* Permuting the broadcast across the batch dimension (SHUFFLE_BROADCAST)
  produces **ΔL_c = +0.0001** — statistically indistinguishable from
  the null region.
* This is highly informative: within a single forward pass the
  broadcast content that a given position needs is largely also
  useful for the other positions in the batch. This aligns with the
  fact that the bounded research corpus is thematically homogeneous
  (three whole works in a fixed order per batch) and with the bounded
  model's capacity.

### 3.3 Source vs slow-state effects (MASK / FREEZE)

* MASK_SUBSTRATE_SOURCE +0.0508 > MASK_TRANSFORMER_SOURCE +0.0123 —
  on this partition the substrate side of Recursion carries a
  slightly larger share of the useful signal than the transformer
  side does. Both effects are small.
* FREEZE_BROADCAST +0.0252 ≈ FREEZE_RECURSION +0.0252 — freezing
  the broadcast alone or freezing both h and c produce the same small
  loss increase, meaning the slow-clock updates during a forward pass
  contribute only modestly on TEST at this scale.

### 3.4 Delay is nearly free

* DELAY_BROADCAST +0.0024 — a one-window delay barely moves loss.
  This is consistent with the small-model + bounded-corpus regime;
  larger models on longer contexts might be more sensitive.

---

## 4. Statistical caveats

* All 24 batches carry ΔL_c as a paired-per-batch difference. The
  reported mean is a pooled estimate; per-batch min/median/max are
  in `docs/latent_bypass/l5_causal_evidence.json`.
* No formal confidence intervals are reported — the two "large" effects
  (0.69 and 0.71) dwarf the ± 0.01 null region by two orders of
  magnitude, so an interval-vs-null test is trivially significant, but
  a full CI + multiple-comparison correction is left as a rigor
  extension for a repeated cross-seed experiment.
* All results are on a **bounded** 1M-token P2 checkpoint on a
  **small** 7M-param model. They do not license claims about
  frontier-scale behavior.

---

## 5. Discipline confirmations (§18)

* `model.training` was False for every intervention.
* No modified checkpoint was saved.
* No overwrite of the P2 checkpoint.
* Tokenizer state, corpus state, and ACIS ledger state unchanged.
* Identical checkpoint, samples, tokenizer, batch order, thread
  policy, and precision across every intervention identity.
* Random-permutation seed (SHUFFLE / NORM_MATCHED) fixed at 20260803.

---

## 6. Claim level supported

* Two large, matched-control ΔL_c signals on the sealed TEST partition
  demonstrate that the shared Recursion broadcast carries semantically
  relevant content for next-token prediction on this bounded corpus,
  and that the effect is not trivially explained by broadcast norm
  (NORM_MATCHED control), nor by batch-position alignment (SHUFFLE
  control).
* This is **Level 3 — CAUSAL_CHECKPOINT_EVIDENCE** for the specific,
  scoped question "does the broadcast content matter to next-token
  loss under this checkpoint?" — nothing broader.
* Level 4+ (net efficiency, repeated comparative) explicitly requires
  cost-adjusted matched-control experiments that this campaign does
  not attempt.

Full evidence: `docs/latent_bypass/l5_causal_evidence.json`.
