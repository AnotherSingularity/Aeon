# Latent Bypass — Theory Lock (L0)

Program: **B — L0 through L11**
Base commit (Program A close): `7d07a44` on `claude/funny-cori-a3k5cf`
Directive: `docs/directives/LATENT_BYPASS_UPGRADE.md`
Status: **L0 — theory locked; scaffolding only.** No causal, efficiency,
or bypass claim has been (or may be) derived from L0.

This document is the immutable frame of reference against which every
subsequent L-series measurement is evaluated. It is written **before**
L1 instrumentation lands so no post-hoc definitional drift can rescue a
weak result.

---

## 1. Statement of the theoretical proposition under test

The Aeon architecture composes two independently updated streams (a
transformer and a substrate) that share a single authorized broadcast
`h_cond` computed by the Recursion module at every `K=16`-step boundary.

The proposition under test is:

> Aeon does not bypass the hidden state. Aeon uses the hidden state to
> bypass a barrier in the visible computational state.

Formally, let `ρ` denote an externally measured visible computational
coordinate (see §3) and let `z = Ψ(r_b)` denote a diagnostic projection
of the hidden Recursion state `r_b` at the `b`-th slow-clock window (see
§4). The proposition asserts that certain trajectories in `ρ` that a
purely-visible-state computation cannot advance efficiently across
become traversable at lower net cost when the hidden coordinate `z`
carries information from earlier windows.

The proposition is not that `z` is a physical quantity, is conserved, or
is causally privileged in any general sense. It is a claim about a
*specific implemented architecture* under specific evaluation
protocols. Every L-series measurement operates inside that scope.

---

## 2. Non-metaphysical stance

The following are explicit disclaimers and hold for every L-tranche:

1. There is no physics claim. `ρ`, `z`, and any potential
   `V(ρ)` used later in this program are **surrogates** for the
   architecture's dynamics. They are the L-series' bookkeeping variables,
   not laws.
2. There is no consciousness claim, subjective-experience claim, or
   claim about internal representations having intrinsic content.
3. Correlation between `z` and future visible improvement is
   **not** treated as evidence of causal contribution. L5 owns the
   causal component.
4. Predictive information in `z` is not evidence of a bypass — L7
   owns the net-efficiency accounting, and a positive predictive result
   with negative net efficiency is not a bypass.
5. Any observed effect must be measurable across more than one
   checkpoint, more than one task class, and more than one dependency
   distance to survive L10 (repetition).

---

## 3. Visible coordinate `ρ`

`ρ` is any externally measurable scalar (or vector) describing progress
of a specific task in the visible substrate of the model. L2 (barrier
registry) enumerates the concrete metrics:

- pre-broadcast negative log-likelihood on the target token,
- pre-broadcast correct-token margin,
- local prediction entropy,
- target rank,
- task progress (task-specific),
- local-state nearest-neighbour similarity,
- dependency distance to the earliest evidence required for the correct answer.

`ρ` operates **on visible data only**. Transformer entropy may be used
by external evaluators to define barriers; it may never enter the
substrate gate at training time (V0.02.02 invariant).

---

## 4. Hidden coordinate `z = Ψ(r_b)`

`r_b` denotes the Recursion state at the `b`-th slow-clock boundary
(one boundary per `K = 16` tokens). `Ψ` is a **declared diagnostic
projection**, not a learned control signal:

- `Ψ_norm(r_b) = ‖r_b − r̄‖₂` — norm coordinate.
- `Ψ_dir(r_b) = v^T (r_b − r̄)` — directional coordinate along a
  linear direction `v` fitted on the calibration partition (typically
  the first PCA direction of calibration-partition `r_b` variance).
- `Ψ_pred(r_b) = f(r_b)` — a predictive coordinate produced by a
  small diagnostic model `f` trained on calibration to predict future
  visible improvement, with capacity documented alongside the result.

The scalar reduction may fail. When it fails, the L-series retains and
reports the vector-level analysis rather than choosing a coordinate
after inspecting held-out results (a form of coordinate p-hacking that
L3 explicitly forbids).

`Ψ` never participates in Aeon's forward pass. It is computed **from**
`r_b` externally by the diagnostic probe layer added in L1 and consumed
only by offline evaluation code.

---

## 5. Existence condition for a bypass

A "bypass" as used by this program is not identified by a single
statistic. It requires **all** of the following, and the report must
explicitly cite each:

1. **Predictive information.** `I(Y_{t+1}; R_b | H_t⁻) > 0` within
   the declared estimator's uncertainty (L8).
2. **Causal contribution.** At least one valid intervention on the
   authorized broadcast pathway materially worsens held-out barrier
   performance (L5, L10).
3. **Barrier selectivity.** The causal effect is stronger inside
   barrier windows than outside (L5, L6).
4. **Net efficiency.** `Δ_capability ≥ C_compute + C_memory +
   C_stability` — the net-benefit accounting after paying for
   entering, maintaining, and consuming the hidden state (L7).
5. **Stability.** Recursion certificate remains valid and full-loop
   evidence shows no uncontrolled amplification (L9).
6. **Repetition.** The effect appears across more than one checkpoint,
   more than one task class, and more than one dependency distance;
   more than one seed where feasible (L10).

Failure of **any** one condition prevents a "demonstrated bypass"
claim (L11 claim-level table).

---

## 6. Path cost and observable consequences

The path cost of using `z` includes:

- `C_compute` — additional forward or diagnostic compute the runtime
  incurs to produce `r_b` and inject `h_cond`.
- `C_memory` — resident bytes of Recursion state and the projection
  matrices that consume it.
- `C_stability` — any amplification budget consumed by cross-cycle
  propagation, measured against the full-loop stability bound (L9).

The observable consequences the L-series can measure are all *visible*:
loss reductions on barrier tasks, throughput deltas, accuracy on
delayed-instruction / long-range-dependency benchmarks. No hidden
consequence is claimed.

---

## 7. Projected-observation inference model

`z` is a projection of `r_b`; it is not `r_b`. The inference model L8
uses treats `Z` as a lossy view of `R`:

- Null model `M₀ : P(Y | H⁻)` — outcome distribution given only the
  visible history.
- Alternative `M₁ : P(Y | H⁻, Z)` — outcome distribution given
  visible history **and** the diagnostic projection.

L8 compares these two models on held-out test data under matched
capacity and shuffled-state controls. A flexible latent model that
wins purely on additional parameters is not evidence of bypass — the
matched-capacity control is explicitly required.

---

## 8. Corpus staging for the L-series (per user directive 2026-07-31)

L0–L2 (plumbing, signal-trace, barrier registry, noninterference tests)
are permitted to operate on the bounded synthetic-English fixture
inherited from W10-11. **No causal, efficiency, likelihood-ratio, or
demonstrated-bypass claim may be derived from that fixture.** The
fixture is used only for deterministic wiring verification and probe
noninterference checks.

L3–L11 (reaction coordinates, interventions, benchmark, efficiency,
inference, stability, integrated decision, closure) require a
small real-English public-domain corpus vendored into the repository
or the local test-data area before execution, with:

- source identity,
- public-domain-status attestation,
- retrieval date,
- exact file digest,
- preprocessing version,
- tokenizer identity,
- partition manifest (train / calibration / validation / held-out
  test), with the held-out test partition sealed until thresholds,
  reaction coordinates, and evidence criteria are locked.

Barrier-specific benchmarks (L6) additionally use curated fixtures for
long-range entity recall, delayed instructions, pronoun resolution,
nested dependencies, contradiction resolution, state tracking,
local-context aliasing, and long-range negation, each with task
identity, dependency distance, barrier location, generator version,
provenance, and partition.

The outbound-network prohibition is **not** lifted for the runtime.
The vendored corpus is authenticated once and then operated on
entirely offline.

---

## 9. Claim ladder

L11 will emit exactly one claim level:

| Level | Description | Corpus |
|-------|-------------|--------|
| 0 | Theory only | Any |
| 1 | Structurally implemented (probe wires cleanly; no runtime change under default) | Synthetic-fixture OK |
| 2 | Observational evidence (predictive information in `z` on real held-out data) | **Real corpus required** |
| 3 | Causal-checkpoint evidence (interventions on the broadcast materially harm barrier performance on one checkpoint) | Real corpus required |
| 4 | Small-scale net-efficiency evidence (net benefit after cost on one checkpoint / one task family) | Real corpus required |
| 5 | Repeated comparative evidence (net benefit reproduces across checkpoints, task classes, dependency distances) | Real corpus required |

Jumping levels is forbidden. The report must state which level was
achieved and why higher levels were not.

---

## 10. Immutability of the theory lock

This document may be extended (appended) but **not** rewritten in ways
that redefine `ρ`, `z`, the existence condition, or the claim ladder
once L1 code lands. Any such redefinition must appear as a new
document with a distinct version and be justified against the outcomes
it would change — the L-series' correctness relies on frame-of-reference
constancy.

---

Signed off: L0 (this commit).
Next: L1 — authoritative signal trace, noninterference tests, and the
diagnostic probe interface on `HybridModel.forward`.
