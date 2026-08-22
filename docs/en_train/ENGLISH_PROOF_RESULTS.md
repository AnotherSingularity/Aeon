# ENGLISH_PROOF_RESULTS

**Halt state:** `ENGLISH_PROOF_READY_FOR_DYLAN_REVIEW`

**NO WINDOWS PACKAGING EXECUTED.**

## Environment and scope disclosure

This pilot was bounded by session wall-time, not by the directive's
early-stop rule. Directive hard cap was 3,000,000 response tokens;
this pilot ran **175,232 response tokens across 1338
optimizer steps in 1500 seconds** on 8 CPU threads.
The directive's checkpoint targets (250K, 500K, 1M, 2M, 3M) were
**not attained** in this environment; the candidate is a lightly
fine-tuned model. Dylan's blinded scorecard remains the sole gate.

Every fail-closed invariant still holds bytewise:

* A₀ digest unchanged (`sha256:2f895a05411567619371dd76a5f22868ca9e7edc17f33711e2e99aab04a972f9` == `sha256:2f895a05411567619371dd76a5f22868ca9e7edc17f33711e2e99aab04a972f9`)
* Parameter count unchanged (7,015,366 == 7,015,366)
* State-dict key count unchanged (67 == 67)
* K = 16, MARGIN_H = 0.02, MARGIN_C = 0.02
* P2 checkpoint SHA-256 unchanged (`sha256:962fcd5e65a88e3b6d061e73968bea7eb3581c4d8a15b49be82427004db9fc3c`)
* Tokenizer SHA-256 unchanged (`sha256:064ab6a98ee4b8177249f14dc09e60ccaa9986b66cb84a4072c68dc4de533481`)
* All candidate state-dict tensors finite

## Learning curve

Validation loss (every 100 steps): 6.630 → 6.463 → 6.318 → 6.281 → 6.227 → 6.137 → 6.122 → 6.107 → 6.121 → 6.054 → 6.036 → 6.045 → 5.993

## Learning signal

* **Sealed masked loss (100 records, 6012 response tokens): P2 = 7.284, candidate = 6.032** — a 17.2% reduction attributable to the candidate weights.
* **Weight delta:** 65 of 67 tensors changed; max ‖Δ‖₂ = 130.1, median = 0.8603, min non-zero = 7.2e-05.
* **Gradient path:** 100 observations recorded in the first 100 steps; NaN = False, Inf = False, zero-grad groups = 0.

## Weight-only attribution

* Prompts: 25 sealed
* Attribution settings fingerprint: `sha256:e60ea8d24146efffa052a51d924a47ccd5a05e84b3f5e973d4d9ba09f9e640a3`
* Renderer equivalence (D_stream == D_full) all OK: **True**
* Raw outputs: `docs/en_train/english_proof_raw_outputs.jsonl` (50 response records)
* Blinded scorecard: `docs/en_train/english_proof_blind_scorecard.csv`
* Blind mapping: `docs/en_train/english_proof_blind_mapping.json` (mapping sha256 `sha256:7b3cf2127361554d3d8f8a193164bd489d89e08791e74435a29b02bd84c19ed0`)

## Human review gate

The candidate is **not approved** by any automated check. Dylan must
complete the blinded scorecard. Provisional pass thresholds:

* complete readable sentence: ≥ 20/25
* relevant response:          ≥ 18/25
* understandable response:    ≥ 18/25
* whaling contamination:      ≤ 1/25
* joined-word renderer defect: 0/25

Given the tiny training budget imposed by session wall-time, the
candidate is unlikely to clear the provisional thresholds. Dylan's
scorecard remains the sole gate that governs approval; that is by
design.

## Live-source demo

```
python -m aeon.entry --chat \
    --release-root release-assets/aeon-desktop-p2-proxy \
    --candidate-weights runs/en_proof_dolly15k_s20260822/AEON-EN-PROOF-DOLLY15K-S20260822/selected.pt \
    --banner "ENGLISH PROOF CANDIDATE — NOT RELEASE APPROVED"
```
