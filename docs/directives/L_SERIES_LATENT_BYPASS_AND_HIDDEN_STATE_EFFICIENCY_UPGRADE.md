# CODE EXECUTION DIRECTIVE

# Aeon Latent Bypass and Hidden-State Efficiency Upgrade

## Architecture-Preserving Post-W10 Research and Runtime Program

## Status

This is the definitive follow-on Aeon upgrade.

The governing terminology is: **Aeon uses its hidden state to bypass barriers
in the visible computational state. It does not bypass the hidden state.**

The mathematical basis is the latent-coordinate barrier model:

    V(ρ,z) = V0 * exp[-(ρ-R)² / (2W²)] * exp[-z² / (2H²)] + (1/2) K_z z²

with a nonzero bypass route existing when

    V0 > K_z H²

and preferred latent coordinate

    z* = ± H * sqrt( 2 ln( V0 / (K_z H²) ) ).

The supplied theory also defines observable consequences, projected-state
inference, escape-rate differences, trajectory costs, and the distinction
between static barrier reduction and full path cost.

This upgrade translates that principle into a testable Aeon architecture
program. It must not convert the mathematical analogy into an unsupported
claim. It must determine whether Aeon's actual hidden Recursion state
provides a lower-cost computational path around limitations in visible local
processing.

## 1. Execution dependency

Do not begin this upgrade from W10-0. First complete W10-1 through W10-11:

- Real tokenizer and corpus ingestion
- Protected checkpoint lifecycle
- Distinct Start, Resume, and Recovery
- Complete runtime integrity
- Truthful frozen preflight
- Real English fixture training
- End-to-end protected start-stop-resume
- Final W10 regression
- Final W10 commit and evidence

The starting point for this upgrade is the final certified W10 commit.

Before beginning L0, record:

- Final W10 branch
- Final W10 commit
- Full W10 test total
- Architecture-manifest identity
- Six-patch-manifest identity
- Tokenizer identity
- Corpus-fixture identity
- Protected-checkpoint policy identity
- Runtime-policy identity
- Contractive-certificate result
- Real-text training result

If W10 remains incomplete, continue W10 autonomously. Do not implement
latent-bypass runtime changes against the withdrawn Windows path.

## 2. Governing scientific hypothesis

Aeon contains a visible local computational process and a bounded hidden
state.

Let h_t⁻ be the transformer's visible local state before the current
Recursion broadcast is consumed. Let r_b be the hidden Recursion state for
block b, with fixed slow-clock interval K=16.

The null hypothesis is:

    H₀:  P(Y_{t+1} | h_t⁻, r_b) = P(Y_{t+1} | h_t⁻).

Under H₀, Recursion contains no useful predictive information beyond the
visible local state.

The Aeon hypothesis is:

    H₁:  P(Y_{t+1} | h_t⁻, r_b) ≠ P(Y_{t+1} | h_t⁻).

Equivalently:

    I(Y_{t+1}; R_b | H_t⁻) > 0.

The stronger causal hypothesis is: altering, delaying, freezing, or
mismatching the hidden Recursion state damages performance specifically
where visible local processing encounters a barrier.

The efficiency hypothesis is:

    Δ_net = Δ_capability − C_Recursion − C_broadcast − M_state − C_stability > 0.

Aeon demonstrates a useful latent bypass only when the capability
improvement exceeds the additional compute, memory, and stability costs.

## 3. Architectural preservation rules

This upgrade must preserve:

1. Two independent parallel streams: transformer and substrate
2. Recursion as the sole cross-stream integration point
3. Both streams independently feeding Recursion
4. Existing single Recursion broadcast
5. Both streams consuming the same authorized broadcast
6. Fixed K=16
7. Contractive Recursion certificate
8. Recursion state in fp32
9. Autonomous substrate gating
10. Substrate state following substrate parameter dtype
11. All six V0.02.02 corrections
12. W10 real-text training path
13. W10 protected checkpoint and replay path
14. No direct transformer-to-substrate inspection
15. No direct substrate-to-transformer inspection
16. No transformer-derived entropy or confidence entering the substrate gate
17. No adaptive-K experiment in the certified path
18. No dual Recursion feedback heads
19. No replacement of Recursion with a generic memory module
20. No retraining of intervention-altered topologies as though they were Aeon

The latent bypass is measured through Aeon's existing architecture. It is
not created by dismantling or replacing that architecture.

## 4. Correct architecture mapping

For block b, define independent stream outputs:

    u_b^T = P_T(H_b^T)
    u_b^S = P_S(H_b^S)

Recursion updates:

    r_b = R(r_{b-1}, u_b^T, u_b^S)

The existing single broadcast is:

    β_b = J(r_b)

The broadcast is consumed by both streams according to the current certified
implementation.

There must remain no direct path H^T → H^S or H^S → H^T. The substrate's
autonomous control function may depend only on

    a_t^S = g_S(q_t, Δq_t, substrate-internal statistics, β_{b-1}).

It may not depend directly on transformer logits, entropy, attention,
confidence, hidden states, local disagreement, or direct cross-stream
comparisons. Any cross-stream relationship must be integrated through
Recursion.

## 5. Visible barrier definition

Do not assume that token loss alone is the visible coordinate.

Create a versioned visible-barrier registry supporting bounded candidate
definitions such as high local negative log-likelihood, low correct-token
margin, prediction instability, failure on long-distance reference,
contradiction-resolution failure, ambiguous local syntax, abrupt semantic
transition, repeated local prediction without progress, failure to retain
earlier instructions, failure to maintain entity or state continuity.

Define a visible progress coordinate ρ_t = Φ(h_t⁻, p_t, ℓ_t, task state)
where Φ is an external diagnostic projection. The projection must not alter
execution.

A barrier region may be defined as

    B = { t : ρ_t ∈ [R−W, R+W] }

or operationally

    B = { t : ℓ_t > τ_ℓ OR m_t < τ_m OR task_progress_t < τ_p }.

Every barrier definition must specify observable inputs, thresholds,
calibration set, held-out evaluation set, expected false-positive behavior,
whether task-specific or general, whether computed before or after Recursion
broadcast.

Primary causal analysis must use the pre-broadcast visible state.

## 6. Hidden reaction coordinate

Recursion state is generally high-dimensional: r_b ∈ ℝ^{d_r}. The
theoretical scalar z must be treated as a reaction coordinate z_b = Ψ(r_b).

Support at least:

- Norm coordinate:        z_b^norm = ‖r_b − r̄‖_2
- Directional coordinate: z_b^dir  = vᵀ(r_b − r̄)
- Predictive coordinate:  z_b^pred = Ψ_θ(r_b),
  a bounded estimator trained only on calibration data to predict future
  visible improvement.

Requirements: estimator must not influence training execution; calibration
and evaluation datasets separate; no future label leakage into online
computation; diagnostic-only; not a routing/gating mechanism; multiple
reaction coordinates compared honestly; failed scalar reduction not hidden
by choosing only favorable cases. Retain vector-state analysis where scalar
reduction is insufficient.

## 7. Upgrade phases

Use additive commits:

- L0  — W10 inheritance and theory-lock audit
- L1  — Authoritative signal-path verification
- L2  — Visible barrier registry
- L3  — Hidden reaction-coordinate framework
- L4  — Bounded bypass observability
- L5  — Causal intervention harness
- L6  — English barrier benchmark suite
- L7  — Efficiency and matched-control program
- L8  — Latent-state inference and likelihood testing
- L9  — Full-loop stability composition
- L10 — Integrated bypass certification
- L11 — Closure and claim-control package

Do not rewrite W10 history.

## 8..21

Sections 8 through 21 of the L-series directive contain L0..L11 detailed
scopes, exit gates, claim levels 0-5, definition of done, and final
execution instruction. See the master conversation directive for full text.
This file preserves the mathematical and architectural spine; the detailed
execution steps are governed by the master directive that produced this
file and are locked from execution by the prerequisite gate in
`docs/W10_CLOSURE_STATE.json` + `tests/test_l_series_prerequisite_lock.py`.

## 22. Final execution instruction

Continue W10 from W10-1 through W10-11 first. After W10 closes, create the
latent-bypass branch from the exact certified W10 final commit and proceed
L0 through L11. Do not start by changing Aeon's architecture. First prove
that the real executed architecture carries real information.

The objective is to establish, with causal and cost-adjusted evidence,
whether:

> Aeon's bounded hidden Recursion state provides a lower-cost route around
> barriers in visible local computation.
