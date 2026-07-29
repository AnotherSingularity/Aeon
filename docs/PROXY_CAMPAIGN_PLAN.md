# Aeon V0.02.03 — Proxy Comparison Campaign Plan

Directive §14: proxy comparisons are planned work, not required before the
primary Aeon training run. Campaign begins only after E0–E6 pass, the main path
is stable, at least one meaningful Aeon checkpoint exists, and a separate
compute budget is approved.

## 1. Preconditions (§14 gate)

- [x] E0 through E6 pass (E7 in progress: this document).
- [ ] The main training path is stable at the primary scale (requires the
      primary Aeon run to have reached at least milestone 3 of §13.4).
- [ ] At least one meaningful Aeon checkpoint exists.
- [ ] Separate compute budget approved.
- [ ] Laptop schedule + storage requirements documented (this file when
      approved becomes those docs).

## 2. Initial campaign scope (§14.1)

The initial campaign is deliberately narrow to make matching feasible:

1. **One small transformer baseline** (~10–50 M params) — Aeon-native transformer
   architecture with adaptive feedback disabled, K removed. Compares against
   Aeon's own transformer side, isolating the contribution of the substrate +
   Recursion path.
2. **One small recurrent baseline** (~10–50 M params) — Aeon's `vector_cell`
   with no Recursion. Isolates whether a "generic RNN" alone matches Aeon's
   integration path.
3. **One small Aeon model preserving the complete topology** (~10–50 M params) —
   the ablation-free scientific control.

No large sweep at start. No dropping to milestone-1 baselines and calling that
a comparison. The claim scope after the initial campaign is strictly small-scale.

## 3. Matching rules (§14.2)

Each entrant must match, as closely as practicable:

| dimension | policy |
|---|---|
| tokenizer | same Aeon tokenizer for all three |
| corpus | same Aeon corpus, same order |
| data ordering policy | fixed seed, single-epoch pack |
| useful-token budget | matched to nearest 1% |
| evaluation set | shared held-out split |
| parameter budget | matched to ±5% |
| sequence-length schedule | identical |
| optimizer class | AdamW; identical weight decay + grad clip |
| precision | bf16 compute, fp32 for the substructures that need it |
| CPU-thread settings | identical |
| checkpoint cadence | identical |
| validation cadence | identical |

Unavoidable differences (there ARE some — e.g. the Aeon model has Recursion +
matrix cell, the transformer baseline does not) are enumerated in the campaign
report, not swept under the rug.

## 4. Compute planning (§14.3)

Before starting proxy training, calculate and record:

- Expected tokens per entrant.
- Estimated steps per entrant.
- Estimated wall-clock duration on the target laptop.
- Disk use per entrant + total.
- Checkpoint count.
- Whether runs are sequential or resumable (both — E3 makes it so).
- Criteria for early termination.

This section is filled in at campaign start; the template stays here.

## 5. Claim boundary (§14.4)

Small proxy results support **only** small-scale evidence. They do NOT prove:

- Aeon beats frontier models.
- Aeon beats every transformer.
- Aeon beats every recurrent architecture.
- Aeon scales better at all sizes.
- Any definitive full-scale superiority.

The published framing after the campaign is:

> Aeon is architecturally efficient by design and has demonstrated measured
> efficiency at small scale under matched or approximately matched CPU
> experiments.

## 6. Repository preservation during the campaign

Nothing in the campaign may:

- Change the K=16 slow-clock cadence for Aeon.
- Alter Recursion, the certificate, or the substrate autonomy rule.
- Reintroduce any V0.02.02 patch defect.
- Instrument beyond the E2 permanent budget in the Aeon entrant.

Each proxy config lives in `configs/proxy_*.yaml` alongside the primary config
so E1's config-invariant tests continue to apply to every entrant.
