# Closure Reference Integration

**Reference.**
`Closure: Mathematics of Adaptive Dynamical Intelligence` — a research
workbook prepared by Horizon Intelligence, Inc., Research Edition,
August 2026. Filename `1fc74382-Closure_Mathematics_of_Adaptive_Dynamical_Intelligence.pdf`,
SHA-256 `sha256:0a5e6308ee5f1ac985690612eb32b7b0361ddc7267b299e7e1a1c7f55e7a4cee`,
64 pages, 1,139,717 bytes. Verified at head `447f0dc`.

**Role and authority.**

* Role: `research_reference`.
* Authority: `non-canonical relative to runtime implementation`.
* The repository is authoritative on Aeon's implemented architecture.
  The PDF is a research reference. A proposed equation in the PDF is
  never automatically promoted to implemented Aeon behavior.

**PDF's own claim boundary (adopted verbatim).**

> Chapters 0 through 7 rely primarily on standard mathematics from linear
> algebra, dynamical systems, control, and adaptive systems. Chapters 8
> and 9 define autonomy and agency operationally; these definitions are
> useful but not unique across science. The AEON translation is a
> proposed research program, not evidence that the architecture already
> possesses autonomy, agency, consciousness, or understanding.

This tranche integrates the PDF as **mathematical honesty and
experimental-design discipline** for the EN-TRAIN program. It does not
implement online reflexivity, new clocks, operator modulation,
closure, autonomy, agency, or self-modification.

---

## 1. Provenance ledger

| Field | Value |
| ----- | ----- |
| filename | `1fc74382-Closure_Mathematics_of_Adaptive_Dynamical_Intelligence.pdf` |
| SHA-256 | `sha256:0a5e6308ee5f1ac985690612eb32b7b0361ddc7267b299e7e1a1c7f55e7a4cee` |
| title | Closure: Mathematics of Adaptive Dynamical Intelligence |
| author | Horizon Intelligence, Inc. |
| date stated in document | Research Edition — August 2026 |
| PDF metadata CreationDate | 2026-08-22T04:23:28Z |
| page count | 64 |
| size (bytes) | 1,139,717 |
| role | `research_reference` |
| authority | `non-canonical relative to runtime implementation` |
| verified at head | `447f0dc` |
| companion machine-readable mapping | `docs/en_train/closure_reference_mapping.json` |

---

## 2. Three indices, one prohibition

Repository binding of the three bookkeeping indices used in EN-TRAIN's
mathematics:

* `i` — token position inside a K-window `w`. Not a clock.
  Bookkeeping over the **FAST CLOCK**'s per-token cadence.
* `w` — slow-clock window index; `W = ceil(T / K)`. Not a clock.
  Bookkeeping over the **SLOW CLOCK**'s window cadence.
* `k` — authorized offline optimizer-update step. Not a clock.
  Bookkeeping over approved offline parameter updates.

**Prohibited mapping.** The PDF's Chapter 10 generic model

```
h_{t+1}    = F_{θ_τ}(h_t, x_t)
θ_{τ+1}    = G(θ_τ, S_τ)
```

is a general adaptive-system model. **Aeon's implemented slow clock is
a RecursionJoiner state tick, NOT a parameter update.** Binding the
PDF's slow θ-update to Aeon's slow clock would falsely claim that Aeon
performs online parameter updates during inference. The repository
does not implement that and the correction order forbids it in this
tranche.

Witnessed absent by:

* `tests/test_desktop_inference_immutability.py`
* Spec: `docs/en_train/EN_TRAIN_CORRECTED_MATHEMATICAL_SPEC.md` §3

---

## 3. Descriptive joint-state view (documentation only)

The PDF's Chapter 7 augmented state `z = (h, m, θ, c, phase, summary)`
is written for the repository as a **descriptive** view of Aeon's
endogenous state. It introduces no runtime variable or update rule:

```
z_{k,w,i} = (
  substrate_state_{w,i},
  h_w, c_w,
  θ_k,
  phase_i,
  window_summary_w,
  existing_stability_diagnostics
)
```

Classification: `descriptive_joint_state_only`. This mapping MUST NOT
be read as a claim of operational closure, autonomy, agency,
consciousness, or general intelligence.

Component bindings are recorded in
`closure_reference_mapping.json.descriptive_joint_state_only`.

---

## 4. Three rates — not three clocks

The PDF (Chapter 10.5) is explicit: *"three rates, not necessarily
three clocks"* and *"the third [rate] need not be implemented as an
independent clock"*. Bindings:

| Rate | Aeon repository binding | Classification | Is a clock? |
| ---- | ----------------------- | -------------- | ----------- |
| 1. fast state evolution | fast clock — per-token `substrate.step` (`aeon/hybrid.py:154-158`) | `implemented` | yes |
| 2. slow RecursionJoiner evolution | slow clock — per-K-window `recursion.step` (`aeon/hybrid.py:175-177`) | `implemented` | yes |
| 3. offline operator learning and/or viability evaluation | offline: `optimizer.step()` at `aeon/en_train/trainer.py:148`; viability: `assert_architecture_invariant`, `assert_native_stability_gate`, `sigma_certificate` at `aeon/en_train/proof.py`; both event/gate-triggered | `offline_training_only` OR `evaluation_only` | **no** |

A periodic architectural clock for rate 3 is **not added**.

---

## 5. Useful PDF mathematics adopted into the spec

Adopted concepts (mapping in `closure_reference_mapping.json.pdf_concepts`):

* **Chapter 5** — recursion is not learning; frozen deployment is not
  parameter-reflexive; learning is retained capability change. The
  spec now separates state evolution from parameter learning using
  this vocabulary and cites `test_desktop_inference_immutability.py`
  as the witness that Aeon's inference is not parameter-reflexive.
* **Chapter 6** — boundedness, stability, invariance, robustness are
  distinct; hard structural constraints beat soft penalties. Aeon
  already treats these separately (finite-parameter, σ-certificate,
  architecture invariant, native stability gate); the spec now
  records the correspondence.
* **Chapter 7** — closure is a complete declared joint-state
  description, not isolation. No hidden oracle. External inputs are
  allowed but must not supply the system's organization. The spec
  adopts these as documentation principles for the descriptive
  joint-state view above.
* **Chapter 11** — long-horizon sensitivity is governed by Jacobian
  products; local stability ≠ global stability. If an adaptive
  operator is ever introduced, its block Jacobian must include
  cross-coupling terms. **Recorded as a prerequisite for any future
  reflexive extension. Not implemented in this tranche.**
* **Chapter 14** — experimental discipline: parameter-matched and
  compute-matched controls; observation-shuffled and frozen-update
  controls (for future reflexivity research only); corpus provenance
  and claim discipline; separate architecture benefit from added
  parameters and added operations. The spec cites these as required
  parts of any English-training claim beyond feasibility.

---

## 6. Mathematics kept as `proposed_future_research` or `prohibited_in_current_tranche`

Classified explicitly and not implemented:

* online θ updates
* fast weights
* hypernetworks
* low-rank online operator changes
* adaptive operator banks
* neuromodulatory parameter changes
* unrestricted self-modification
* projected online reflexivity
* internally generated valuation
* autonomy (PDF Chapter 8 sense)
* agency (PDF Chapter 9 sense)

The corresponding null claims are recorded in
`closure_reference_mapping.json.consequences_for_current_tranche.must_not_implement`.

---

## 7. Interaction with earlier EN-TRAIN evidence

* `docs/en_train/EN_TRAIN_ARCHITECTURE_FREEZE.json` — A₀ fingerprint,
  P2 hash, tokenizer hash: unchanged by this tranche.
* `docs/en_train/EN_TRAIN_CLOCK_MAPPING.md` +
  `en_train_clock_mapping.json` — canonical fast/slow clock inventory:
  unchanged.
* `docs/en_train/EN_TRAIN_CORRECTED_MATHEMATICAL_SPEC.md` — updated
  in a separate commit to add a Closure Reference Integration
  section (three-rate view, descriptive joint state, prohibitions,
  adopted PDF mathematics).
* `docs/en_train/en_train_repository_symbol_mapping.json` — updated
  with a `closure_reference` companion pointer.
* `docs/en_train/en_train_equation_bindings.json` — unchanged. Every
  equation in the spec is still bound to a repository source.

---

## 8. Documentation drift already recorded

Two stale `docs/TOPOLOGY_MAP.md` line-number citations for the
window loop and inject sites remain **unrepaired** per the correction
order's prohibition on modifying canonical clock references during
this tranche. This tranche does not touch that drift. It is recorded
in the earlier commit `7b460e5` and in `EN_TRAIN_CORRECTED_MATHEMATICAL_SPEC.md` §7.

---

## 9. Bottom line

The Closure PDF strengthens the mathematical honesty and experimental
discipline of the English-training program. It is **not** used to
rewrite Aeon into the PDF's proposed future architecture. The repository
remains authoritative; the PDF remains a research reference; the
correction order's prohibition on online reflexivity, new clocks,
operator modulation, closure, autonomy, agency, or self-modification
in this tranche is preserved.

Training remains halted at `AWAITING_OFFLINE_CORPUS_SOURCES`.
