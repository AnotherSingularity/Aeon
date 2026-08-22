# EN-TRAIN — Corrected Mathematical Specification

**Head:** `447f0dc` (evidence commit `7b460e5` — dual-clock inventory).
**Scope:** authorized offline English training (Stages 1 and 2) and its
evaluation infrastructure. Does not extend to any inference-time
parameter modification, any architectural change, any clock change, or
any change to Aeon's native forward semantics.

This document supersedes the earlier attempt that tried to bind
`C_1`/`C_2` to time indices. Aeon's **two architectural clocks are
pre-existing in the repository**: the FAST CLOCK (per-token substrate
recurrence) and the SLOW CLOCK (once-per-K-window RecursionJoiner
tick, shared broadcast, and transformer inject). Repository evidence
is anchored in:

* `docs/en_train/en_train_clock_mapping.json`
* `docs/en_train/EN_TRAIN_CLOCK_MAPPING.md`
* `docs/en_train/en_train_repository_symbol_mapping.json`

The bindings and equations below are read out of the repository at
`447f0dc`; nothing here changes Aeon's forward semantics. Where the
correction order requires an update operator, the repository already
contains the operator (`torch.optim.AdamW` via
`aeon.en_train.trainer.train_one_step`); no new learned operator is
introduced.

---

## 1. Indices, and what they are NOT

| Symbol | Meaning | NOT one of Aeon's architectural clocks |
| ------ | ------- | -------------------------------------- |
| `i`    | token position inside a K-window `w`, `i ∈ [start, end)`, `start = w·K`, `end = min((w+1)·K, T)` | correct — `i` is a bookkeeping index over the FAST CLOCK's fast-tick cadence |
| `w`   | slow-clock window index, `w ∈ [0, W)` with `W = ceil(T / K)` | correct — `w` is a bookkeeping index over the SLOW CLOCK's window cadence |
| `k`   | authorized offline optimizer-update step | **not a clock** — `k` counts approved offline parameter updates during EN-TRAIN training, is applied outside the forward path, and never appears in Aeon's inference |

The FAST CLOCK and SLOW CLOCK are Aeon's two architectural clocks (see
`EN_TRAIN_CLOCK_MAPPING.md`). `i`, `w`, and `k` are **bookkeeping
indices used to write down the equations**. They are not the clocks.

The correction order's directive is enforced literally: EN-TRAIN does
not define, add, rename, or reinterpret Aeon's dual clocks. It only
labels the two clocks the repository already establishes.

---

## 2. Aeon's native forward process (unchanged)

Aeon's native forward, at any parameter state `θ_k`, is the single
existing call

```
HybridOutput = HybridModel.forward(input_ids, attention_mask=…, labels=…)
```

at `aeon/hybrid.py:112-284`. The **FAST CLOCK** ticks once per input
token inside each K-window; the **SLOW CLOCK** ticks once per K-window
at the window boundary and writes back into the transformer via the
inject path. `K = 16` at `aeon/hybrid.py:68,78` and is pinned by every
config and by `docs/PRESERVATION_MANIFEST.md:21`.

### 2.1 Abstract structure

Following the correction order's instruction to express the forward
process initially in a clock-agnostic form:

```
(S_{n+1}, z_n) = F_{θ_k}(S_n; FAST_CLOCK, SLOW_CLOCK)
```

where:

* `F_{θ_k}` is `HybridModel.forward` at the parameter state after
  optimizer update `k`;
* `S_n = (substrate_state, recursion_h, recursion_c)` is the native
  carried internal state (repository containers per
  `en_train_repository_symbol_mapping.json`);
* `z_n` is `HybridOutput.logits[n] ∈ ℝ^{V}` with `V = 16000`;
* `n` is a bookkeeping index over Aeon's native forward process — it
  is not a clock;
* `FAST_CLOCK` and `SLOW_CLOCK` are the two pre-existing architectural
  clocks anchored in the repository (fast clock: `aeon/hybrid.py:154-
  158`; slow clock: `aeon/hybrid.py:146-178` + `aeon/recursion.py:148`).

Aeon does not persist `S` across forward calls; each
`HybridModel.forward` call reinitializes substrate + recursion state
at `aeon/hybrid.py:141-142`. That is a pre-existing Aeon property and
EN-TRAIN does not change it.

### 2.2 Repository-anchored equations

The equations below are read out of the current implementation. They
are stated in the form the repository executes; they are not
simplified, generalized, or "improved". Line references anchor at
`447f0dc`.

**FAST CLOCK — per-token substrate recurrence** (`aeon/hybrid.py:151-
158`, `aeon/substrate/matrix_cell.py::MatrixCell.step`):

```
x_{w,i}                  = emb_proj(emb_{w,i}) + cond_proj(h_{w-1})
r_{w,i}, substrate_{w,i} = SubstrateStep(x_{w,i}, substrate_{w,i-1})
```

with the pre-existing per-window boundary reinitialization from
`aeon/hybrid.py:141`.

**Window aggregation** (`aeon/hybrid.py:160-161`):

```
s_w = s_proj( mean_{i ∈ [start_w, end_w)} r_{w,i} )
```

**Transformer read at the K-th position** (`aeon/hybrid.py:162`):

```
t_w = t_all[:, end_w - 1, :]
```

with `t_all = transformer.read(transformer.hidden_states(...))` at
`aeon/hybrid.py:136-138`.

**Optional embedding-mean input** (`aeon/hybrid.py:164`):

```
e_w = mean_{i ∈ [start_w, end_w)} emb_{w,i}    if recursion.use_embedding_input else ∅
```

**SLOW CLOCK — RecursionJoiner tick** (`aeon/hybrid.py:175-177`,
`aeon/recursion.py:148-169`):

```
(h_w, c_w) = RecursionStep(s_w, t_w, stopgrad(h_{w-1}), stopgrad(c_{w-1}), e_w)
```

where `stopgrad(·) = ·.detach()` at the slow-clock boundary is the
pre-existing truncated-BPTT design (D2) at `aeon/hybrid.py:176,178`.

Expanded from `aeon/recursion.py:148-169`:

```
c_w = (1 − λ) · c_{w-1} + λ · tanh( h_{w-1} · W_cᵀ )
h_w = tanh( W_s · s_w + W_t · t_w + optional(W_e · e_w) + h_{w-1} · W_hᵀ + c_w )
```

**Broadcast and transformer inject** (`aeon/hybrid.py:150,158,254-256`):

```
h_cond_w = h_{w-1}                          # held broadcast for window w
inject_signal[·, i, ·] = h_cond_w           for every i ∈ [start_w, end_w)
injected = transformer.inject(hidden, inject_signal)
logits  = transformer.logits(injected) = lm_head(hidden + γ · write_proj(inject_signal))
```

**Substrate boundary detach** (`aeon/hybrid.py:178`):

```
substrate_{w, end_w} ← detach( substrate_{w, end_w} )
```

**Binding σ certificate** (`aeon/recursion.py:22-23,100-101`):

```
σ_max(W_h) < MARGIN_H
σ_max(W_c) < MARGIN_C
```

**Clock ratio** (`aeon/hybrid.py:68,78`; every config; P-K16):

```
K = 16
W = ceil(T / K)
```

None of the above is modified by EN-TRAIN. The equations are stated
exactly as the repository implements them.

---

## 3. Offline English learning (separate from the forward process)

Offline learning is a **strictly outer loop** around the frozen native
forward process. It runs only during authorized training, never during
inference.

### 3.1 Loss L_k

Both losses consume `HybridOutput.logits` directly with nothing
inserted between logits and loss (§5 of the correction order).
Repository sources:

* `L_G` — `aeon.en_train.losses.general_english_loss`
  (`aeon/en_train/losses.py:48-58`)
* `L_C` — `aeon.en_train.losses.conversational_loss`
  (`aeon/en_train/losses.py:64-80`)

Both dispatch through `masked_next_token_loss`:

```
L_k(θ_k) = − ( sum_{(b,t) ∈ M_k}  log softmax(z_{b,t}(θ_k))[y_{b,t+1}] )
             / |M_k|
```

where the mask `M_k` is:

* `L_G`: `M_k[b,t] = attention_mask[b, t+1]`
* `L_C`: `M_k[b,t] = attention_mask[b, t+1] · response_mask[b, t+1]`

### 3.2 Gradient g_k

```
g_k = ∇_{θ_k} L_k(θ_k)
```

Repository site: `loss.backward()` at
`aeon/en_train/trainer.py:146`. Gradient-path observation over the
first 100 updates: `aeon.en_train.proof.observe_gradient_path` and
`assert_gradient_path_over_100_steps` (`aeon/en_train/proof.py:110-
162`).

### 3.3 Gradient control

The gradient-clipping map is exactly (`aeon/en_train/trainer.py:117-
127`):

```
ḡ_k = g_k · min( 1, c / max( ‖g_k‖_2 , ε ) )      with c = 1.0
```

### 3.4 Stability diagnostics q_k

`q_k` is the pre-existing native diagnostics vector — nothing is
invented. Repository sources:

* `aeon.en_train.proof.sigma_certificate` (MARGIN_H, MARGIN_C
  bookkeeping, `aeon/en_train/proof.py:229-237`), backed by
  `aeon.recursion.RecursionJoiner.audit` at `aeon/recursion.py:173-184`
* `aeon.en_train.proof.check_finite_state_dict`
  (`aeon/en_train/proof.py:219-226`)
* `aeon.en_train.proof.assert_architecture_invariant`
  (`aeon/en_train/proof.py:65-80`)

### 3.5 Update operator G_existing

The correction order (§4) permits the existing approved optimizer
path when no native learned update operator exists. Repository search
confirms none exists in `aeon/*`; every training entry point uses
`torch.optim.AdamW`. Therefore:

```
θ_{k+1} = G_existing( θ_k, ḡ_k, q_k )
```

is instantiated as `optimizer.step()` at
`aeon/en_train/trainer.py:148` after the `_apply_grad_clip` call and
after every native diagnostic has been evaluated at the gate
(`aeon/en_train/trainer.py:381-397`). No new learned operator is
introduced.

### 3.6 Update-time invariants

For every accepted optimizer update `k → k+1`, EN-TRAIN requires
(§2 of the correction order):

```
θ_{k+1} ≠ θ_k                                   ← Δ_weights > 0
architecture( F_{θ_{k+1}} ) = architecture( F_{θ_k} )     ← Δ_architecture = 0
```

Repository enforcement:

* Δ_weights > 0 witnessed by
  `aeon.en_train.proof.compute_weight_delta` at end of each stage
  (`aeon/en_train/trainer.py:421-435`).
* Δ_architecture = 0 enforced by
  `aeon.en_train.proof.assert_architecture_invariant` at every
  learning-curve gate (`aeon/en_train/trainer.py:382-385`), comparing
  the current fingerprint digest against
  `PROTECTED_A0_DIGEST` and the parameter count against
  `PROTECTED_TOTAL_PARAMETERS` (`aeon/en_train/__init__.py`).

### 3.7 What is deliberately absent

No online / inference-time parameter modification path exists in the
repository, and the correction order (§7) forbids implementing one.
The corresponding null claim is captured in
`en_train_repository_symbol_mapping.json.not_present_in_repo_and_not_invented`
and is now also enforced by a test
(`tests/test_desktop_inference_immutability.py`, §5 of this
document).

---

## 4. Separation of concerns

| Concern | Where it lives | Whose responsibility |
| ------- | -------------- | -------------------- |
| Aeon's native forward process (FAST + SLOW clocks) | `aeon/hybrid.py`, `aeon/recursion.py`, `aeon/substrate/*` | pre-existing Aeon architecture; **frozen for EN-TRAIN** |
| Offline English learning (`L_G`, `L_C`, `g_k`, `ḡ_k`, `q_k`, `G_existing`) | `aeon/en_train/*` | EN-TRAIN infrastructure; runs only in authorized training scripts |
| Rendering | `aeon/desktop/runtime.py` render path | corrected by EN-TRAIN-1 (renderer fix); not touched here |
| Inference θ-immutability | `aeon/desktop/runtime.py` and every generation caller | enforced by `tests/test_desktop_inference_immutability.py` |

The three concerns must remain distinguishable. Recursion is
architectural (frozen). Offline learning is bookkeeping over `k`
(runs only when authorized). Rendering is a display-only projection
of `logits` into visible text.

---

## 5. Machine-readable equation bindings

Every equation in this document is bound to a repository source in
`docs/en_train/en_train_equation_bindings.json`, along with the
symbol → repo mapping already recorded in
`docs/en_train/en_train_repository_symbol_mapping.json`.

Consumers that need to verify a claim against the runtime should read
the JSON files; this Markdown file is the narrative view of the same
bindings.

---

## 6. What this document does not authorize

* Any change to `K`, MARGIN_H, MARGIN_C, `h`/`c` state dimensions,
  substrate state, the FAST-CLOCK cadence, the SLOW-CLOCK cadence,
  the truncated-BPTT detach boundaries, the conditioning path
  (`h_cond → cond_proj`), the aggregation path (`mean_i r → s_proj`),
  or the injection path (`transformer.inject`).
* Any modification of protected artifacts (P2 checkpoint, tokenizer,
  release manifest, architecture fingerprint A₀).
* Any inference-time modification of `θ`.
* Beginning English training. EN-TRAIN infrastructure holds at
  `AWAITING_OFFLINE_CORPUS_SOURCES` per
  `docs/en_train/EN_TRAIN_CORPUS_INTAKE_CONTRACT.md`.

---

## 7. Closure Reference Integration

Provenance: `docs/en_train/CLOSURE_REFERENCE_INTEGRATION.md` +
`docs/en_train/closure_reference_mapping.json`. The Horizon Intelligence
reference workbook *Closure: Mathematics of Adaptive Dynamical
Intelligence* (SHA-256
`sha256:0a5e6308…5e7a4cee`, 64 pp., Research Edition August 2026) is
integrated as a research reference — `role: research_reference`,
`authority: non-canonical relative to runtime implementation`.

### 7.1 Three-index correction (adopted verbatim from Section 1)

Indices `i`, `w`, `k` are bookkeeping only. `k` is not a clock.

### 7.2 Prohibited PDF-to-repository mapping

The PDF Chapter 10 generic model

```
h_{t+1}  = F_{θ_τ}(h_t, x_t)
θ_{τ+1}  = G(θ_τ, S_τ)
```

is a general adaptive-system model. **Aeon's implemented slow clock is
a RecursionJoiner state tick, not a parameter update.** The mapping

> Aeon slow clock ≡ parameter update `θ_{τ+1}`

is **prohibited**. Binding the PDF's slow θ-update to Aeon's slow
RecursionJoiner clock would falsely claim Aeon performs online
parameter updates during inference; the repository does not implement
that and the correction order forbids it in this tranche. Witnessed
absent by `tests/test_desktop_inference_immutability.py`.

### 7.3 Descriptive joint-state view

The PDF Chapter 7 augmented state is written for the repository as a
non-executable descriptive mapping:

```
z_{k,w,i} = ( substrate_state_{w,i},
              h_w, c_w,
              θ_k,
              phase_i,
              window_summary_w,
              existing stability diagnostics )
```

Classification: `descriptive_joint_state_only`. This mapping introduces
no runtime variable, no update rule, and no operational-closure claim.
It MUST NOT be read as evidence of operational closure, autonomy,
agency, consciousness, or general intelligence.

### 7.4 Three rates, not three clocks

Per PDF Chapter 10.5 (*"three rates, not necessarily three clocks"*):

1. **Fast state evolution** — FAST CLOCK, `aeon/hybrid.py:154-158`.
2. **Slow RecursionJoiner evolution** — SLOW CLOCK,
   `aeon/hybrid.py:175-177`.
3. **Offline operator learning and/or viability evaluation** —
   offline: `optimizer.step()` at `aeon/en_train/trainer.py:148`;
   viability: `assert_architecture_invariant`,
   `assert_native_stability_gate`, `sigma_certificate` at
   `aeon/en_train/proof.py`. Both are event/gate-triggered, not
   periodic. **No third architectural clock is added.**

### 7.5 Adopted PDF mathematics (motivation only)

* **Ch 5** — recursion ≠ learning; frozen deployment is not
  parameter-reflexive. Motivates the spec's separation of state
  evolution from parameter learning.
* **Ch 6** — boundedness, stability, invariance, robustness are
  distinct; hard structural constraints beat soft penalties. Aeon
  already treats these separately.
* **Ch 7** — closure = complete declared joint-state description,
  not isolation; no hidden oracle; external inputs allowed but must
  not supply the system's organization.
* **Ch 11** — long-horizon sensitivity governed by Jacobian
  products; **if** an adaptive operator is ever introduced, its
  block Jacobian must include cross-coupling terms. Recorded as a
  prerequisite for any future reflexive extension.
* **Ch 14** — parameter-matched and compute-matched controls;
  observation-shuffled and frozen-update controls (for future
  reflexivity research only); corpus provenance and claim discipline;
  separate architecture benefit from added parameters and added
  operations. Adopted as experimental discipline for EN-TRAIN.

### 7.6 Mathematics kept as `proposed_future_research` or `prohibited_in_current_tranche`

Explicitly not implemented in this tranche:

online θ updates · fast weights · hypernetworks · low-rank online
operator changes · adaptive operator banks · neuromodulatory parameter
changes · unrestricted self-modification · projected online
reflexivity · internally generated valuation · autonomy · agency.

Full list and per-item repository classification in
`closure_reference_mapping.json`.

### 7.7 Non-claims (adopted verbatim from PDF Chapter 13.6)

The following MUST NOT be inferred from the descriptive joint-state
view or from any other equation in this spec:

* consciousness;
* general intelligence;
* self-awareness in the ordinary human sense;
* semantic understanding across unrestricted domains;
* autonomy outside declared viability criteria;
* safety under untested conditions.

---

## 8. Documentation-vs-runtime discrepancies (unrepaired, reported)

Two stale line-number citations in `docs/TOPOLOGY_MAP.md` were
reported at `EN-TRAIN-3`:

* Cites `hybrid.py:127-151` for the window loop (actual `146-178`).
* Cites `hybrid.py:139` for `inject_cols.append` and `:154` for
  `transformer.inject` (actual `:158` and later, respectively).

Both are stale line numbers only; the described content matches
runtime and is confirmed by
`tests/test_recursion_topology.py::test_recursion_step_called_once_per_window`.
Not repaired in this tranche per the correction order's prohibition
on modifying canonical clock references during English training;
recorded as documentation-only follow-up work.
