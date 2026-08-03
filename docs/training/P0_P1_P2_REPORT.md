# Training Report — P0 / P1 / P2

**Corpus:** AEON-LBC-1, train partition (`research-data/AEON-LBC-1/processed/train.jsonl`)
**Tokenizer:** `research-data/AEON-LBC-1/tokenizer/aeon-lbc1.model`
              (unigram, 16000 pieces, byte-fallback, sha256:064ab6a9…)
**Model:** HybridModel, transformer 256/4/4, h_rec=128, **K=16 fixed**, 7.015 M params
**Device / dtype:** CPU / fp32
**Seed:** 20260731 (as declared in `configs/latent_bypass/aeon_lbc1_proxy.yaml`)
**Runner:** `scripts/run_pipeline_stage.py --stage {P0,P1,P2}`

Sealed test (PG-1661) is never opened by any of these stages.

---

## 1. Aggregate

| Stage | Budget (useful tokens) | Actual   | Steps | Wall (s) | Tokens/sec | First loss | Last loss | Min loss |
| ----- | ---------------------- | -------- | ----- | -------- | ---------- | ---------- | --------- | -------- |
| P0    |   16,384               |   16,384 |    8  |    15.9  |   1030     | 9.7562     | 8.9757    | 8.9757   |
| P1    |  262,144               |  262,144 |  128  |   134.6  |   1948     | 9.7562     | 6.5049    | 6.4762   |
| P2    | 1,048,576              | 1,048,576|  512  |   509.0  |   2060     | 9.7562     | 5.8979    | 5.4501   |

All three stages ran on the same seed with the same corpus + tokenizer + model
config. Loss trajectory shows real learning signal, not synthetic noise:
9.76 → 5.90 over the 512-step P2 run (a factor of 40 in log-likelihood).

---

## 2. Corpus coverage (P2)

Train partition tokenizes to roughly 460k tokens at unigram-16k. P2's
budget of 1,048,576 useful tokens therefore requires ~2.3 epochs
through TRAIN. `pack_batches` in `scripts/run_pipeline_stage.py` wraps
the token stream when it reaches the end so the full budget is
covered. Multi-epoch on a bounded corpus is intentional for the
bounded-research configuration (§9); no partition ever crosses
partition boundaries.

---

## 3. ACIS matched-trial equivalence (§10, §13)

At the ending state of each stage, one held-out batch was fed
through `HybridModel.forward` three ways and the resulting logits
digested:

| Stage | OFF loss | OBSERVE loss | BUCKET loss | OFF ≡ OBSERVE logits | OFF ≡ BUCKET logits |
| ----- | -------- | ------------ | ----------- | -------------------- | ------------------- |
| P0    | 8.8830   | 8.8830       | 8.8830      | ✔ (byte-identical)   | ✔ (byte-identical)  |
| P1    | 6.9378   | 6.9378       | 6.9378      | ✔                    | ✔                   |
| P2    | 5.9861   | 5.9861       | 5.9861      | ✔                    | ✔                   |

OFF is byte-identical to the shuttle-absent forward (per ACIS-3 test
`test_shuttle_none_produces_byte_identical_forward`). OBSERVE and
BUCKET do not perturb logits or loss at any stage. This is the
"no accidental detach / no duplicate broadcast" gate from §10.

---

## 4. Architectural invariants held across all three stages

* `K = 16` fixed — confirmed by `HybridModel.K == 16` at ending state of every stage.
* One broadcast per K-boundary — enforced by `HybridModel.forward` shape;
  no code path spawns a second broadcast.
* Recursion state fp32 — `recursion.step` receives `.float()`-cast inputs.
* Substrate autonomous — `substrate.step(x_i)` still gets its own input.
* ACIS default OFF during canonical training — `HybridModel.forward`
  was called without a `shuttle` kwarg for every training step.
* No sealed test access — `pack_batches` reads only `train.jsonl`.
* Six V0.02.02 patches intact — `tests/test_six_patches.py` still passes.

---

## 5. Checkpoint identity (P2)

* Path: `runs/aeon_lbc1_P2/final.pt` (git-ignored; large binary)
* SHA-256 (bytes on disk): recorded in `docs/training/p2_evidence.json`
* Contents: `model_state_dict`, `stage="P2"`, `useful_tokens=1048576`,
  `n_params=7015366`, `vocab_size=16000`, `K=16`, `seed=20260731`
* This checkpoint becomes the **fixed experimental basis for L3 → L5**.
  No further training happens against it.

---

## 6. What these numbers do NOT claim

* P2 is a **bounded research checkpoint** — 1M tokens on a 7M-parameter
  small model. Loss ~6 is far from any frontier and does not claim
  frontier-scale capability.
* Multi-epoch training over a bounded literary corpus is chosen for
  reproducibility and containment; it is not evidence of net-efficient
  training under a matched compute budget.
* ACIS OBSERVE and BUCKET equivalence is on **logits + loss** at the
  ending checkpoint. Full workload-level overhead certification is
  reported separately under `docs/acis/ACIS_WORKLOAD_CERTIFICATION.md`.

Full evidence for each stage lives in `docs/training/{p0,p1,p2}_evidence.json`.
