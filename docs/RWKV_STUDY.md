# RWKV State-Propagation Study

A reference study of RWKV's state-propagation design, read against Aeon's
current architecture, to inform Aeon's next-generation information-flow
decisions.

**Source studied:** [`BlinkDL/RWKV-LM`](https://github.com/BlinkDL/RWKV-LM).
A focused, text-only subset of the studied files is vendored under
[`../reference/RWKV-LM/`](../reference/RWKV-LM/) so the file/line citations
below resolve in-repo (the full upstream tree carries ~5 MB of images plus many
model generations that the study did not need — see that folder's
`PROVENANCE.md`).

**Aeon source referenced:** the `aeon/` package handed over for this work
(`model.py`, `block.py`, `recursion.py`, `config.py`).

**Files read, top to bottom:**

| File | What it gave us |
|---|---|
| `RWKV-v5/rwkv_v6_demo.py` | The canonical RNN-form + GPT-form of RWKV-6, with the author's own inline derivation of the WKV operator. |
| `RWKV-v5/src/model.py` | Training-form (GPT-mode) implementations of v5 (`x052`), v6 (`x060`, and the `_state` / `a`/`b`/`c` variants) and v7 (`x070`). |
| `RWKV-v7/rwkv_v7_demo_rnn.py` | The RNN-form of RWKV-7 "Goose" — the clearest statement of the v7 state object and update. |
| `RWKV-v5/cuda/wkv7_cuda.cu` | The fused WKV-7 forward/backward kernel — the actual recurrence as executed. |
| `README.md` | The architectural framing ("Parallelizable RNN with Transformer-level performance", "100% attention-free", "infinite ctxlen"). |
| `RWKV-8.md` | The forward-looking design direction (the "state ladder", larger/smaller/mixed/fancy state). |

> **One attribution correction up front.** The brief frames the feature
> "state of the last layer of token N → state of the first layer of token
> N+1" as the *RWKV-7 advancement*. That specific cross-layer/cross-token
> state mixing is **not** a shipped RWKV-7 feature — it is described in
> `RWKV-8.md` §3 ("Mixed State") as a *future* direction ("A depth-L model
> becomes a depth-2L model after a step of this"). What RWKV-7 actually ships
> is (1) a matrix-valued, data-dependent state **transition** (a generalized
> delta rule) and (2) the **value-residual** (`v_first`) cross-layer shortcut.
> Both are covered below, and the RWKV-8 idea is covered as roadmap.

---

## a) How state propagates in RWKV

### What "state" is at each block

RWKV is built from a stack of identical residual **blocks**, each containing a
**time-mix** sub-layer (the attention replacement) and a **channel-mix**
sub-layer (the FFN replacement):

```
x => emb => ln0
  => x + att(ln1(x))   # time-mix  (RWKV_Tmix)   -- carries cross-token state
  => x + ffn(ln2(x))   # channel-mix (RWKV_CMix) -- FFN, mostly within-token
  => ... (L blocks) ...
  => ln_out => head => logits
```

The recurrent **state** lives in the time-mix sub-layer. Per block it is:

* **RWKV-6** (`rwkv_v6_demo.py` time_mixing, and `model.py:325` `RWKV_Tmix_x060`):
  * a **matrix WKV state** `S` of shape `(H, N, N)` per layer, where `H` =
    number of heads and `N` = head size = 64. It is split per head; each head
    owns a `64×64` matrix.
  * plus a **token-shift buffer**: the previous token's input vector
    (shape `C = n_embd`) used by the time-shift.
* **RWKV-7** (`rwkv_v7_demo_rnn.py`, init at lines 285–288): per layer, exactly
  **three** tensors:
  * `state[i*3+0]` — time-mix token-shift buffer, shape `(C,)`.
  * `state[i*3+1]` — the **WKV matrix state**, shape `(H, N, N)`, kept in fp32.
  * `state[i*3+2]` — channel-mix token-shift buffer, shape `(C,)`.

**Semantics.** The matrix state is an *associative memory* / linear-attention
accumulator: it stores a (decayed) sum of outer products of keys and values,
`S ≈ Σ_t decay · (v_t ⊗ k_t)`. A query-like vector `r` reads from it by matrix
multiply. The author's own derivation in `rwkv_v6_demo.py` (lines 205–221)
spells this out:

```
S_t = u·A_t + A_{t-1} + w_{t-1}·A_{t-2} + w_{t-1}·w_{t-2}·A_{t-3} + ...
x_t = r_t @ S_t          # read out
A_t = k_t ⊗ v_t          # the new "memory" written this step
```

The total inference state is `f(B, H, N, L, Q)` — independent of sequence
length `T`. This is the headline property: **constant-size state, no
KV-cache** (`RWKV-8.md` line 27: "RNN statesz = f(B,H,N,L,Q). Transformer
statesz = f(B,T,H,N,L,Q).").

### How state passes between tokens (time-mix)

This is the core recurrence. Per head, per step:

**RWKV-6** (`rwkv_v6_demo.py` lines 194–200):
```
a = k ⊗ v                 # outer product, the new association (N×N)
x = r @ (u·a + s)         # read: current-token bonus u plus accumulated state
s = a + w · s             # write: decay old state by per-channel w, add new a
```

* `s` is the carried state; it **decays multiplicatively by `w`** (a per-channel
  vector in `(0,1)`) and **accumulates** the new outer product each token.
* `u` (`time_faaaa`) is a per-channel **"bonus"** that gives the *current*
  token full, undecayed weight — it lets the model treat "now" specially
  without routing it through the decay.

**RWKV-7** (`rwkv_v7_demo_rnn.py` lines 143–146; kernel `wkv7_cuda.cu` lines
37–41) generalizes the *transition* from a diagonal decay to a full
data-dependent matrix:
```
vk = v ⊗ k                                   # new association  (N×N)
ab = (-kk) ⊗ (kk·a)                           # data-dependent removal/transition
state = state · diag(w) + state @ ab + vk     # evolve
out   = state @ r                             # read
```
* `state @ ab` is the new piece: a **rank-structured, value-dependent
  transition matrix** that can *remove or overwrite* previously stored
  associations, not merely shrink them uniformly. This is a **generalized
  delta rule** (the DeltaNet / test-time-training family).
* `a = sigmoid(...)` is explicitly called the **"in-context learning rate"**
  (`model.py:879`). `kk` is an L2-normalized "removal key". Together they let
  the state behave like an online learner that updates a small associative
  memory as it reads the sequence.
* `w = exp(-0.606531 · sigmoid(...))` (or `exp(-exp(·))` in the naive form) —
  per-channel decay, still strictly in `(0,1)`.

So between tokens, RWKV carries a **bounded, fixed-size matrix per head** that
is decayed and updated every step. The cost is `O(1)` per token, `O(T)` per
sequence — and because the update is a linear/associative scan, it is **also
parallelizable for training** ("GPT mode").

### How state passes between layers (channel-mix) — with a correction

The brief asks how state "passes between layers (channel-mix mechanism)". The
precise answer is: **in RWKV, inter-layer information flows through the
residual stream `x`, exactly as in a transformer** (`x = x + att(...)`,
`x = x + ffn(...)`). The **channel-mix is not the inter-layer state carrier** —
it is RWKV's **FFN replacement** (`rwkv_v6_demo.py` lines 83–91):

```
xx = time_shift(x) - x            # "previous token minus this token"
xk = x + xx · time_maa_k          # token-shifted input
xr = x + xx · time_maa_r
k  = relu(key(xk)) ** 2           # squared-ReLU activation
return sigmoid(receptance(xr)) · value(k)   # extra sigmoid gate
```

Channel-mix mixes **across channels within a (token-shifted) position** and
applies a receptance gate. It *does* carry a one-token memory through its own
token-shift buffer (`state[i*3+2]`), but that is a tiny local context, not the
sequence-spanning state. **The sequence state lives in time-mix; layers are
stacked through the residual stream.**

(There *is* a real cross-layer shortcut in RWKV-7 — the value residual — see
the RWKV-7 subsection below.)

### The role of learnable per-channel time-decay

Decay `w` is **per-channel and learnable**, and this is load-bearing in two
ways:

1. **Multi-timescale memory.** Each channel decays at its own rate. The init
   deliberately spreads decay across a spectrum from fast to slow
   (`rwkv_v6_demo.py` lines 354–357; `model.py:815`
   `www[n] = -6 + 6·(n/(C-1))**(...)`). Fast channels capture local structure;
   slow channels carry long-range information. The model chooses *which channel
   to write information into* to choose *how long it persists*.
2. **Parallelizable training.** The README (line 418) makes the key point:
   because the base decay is **data-independent per channel**, RWKV does not
   need per-step gates like a classic LSTM/GRU — "you simply move the
   information from a W-0.8-channel to a W-0.5-channel". A data-independent
   linear recurrence can be evaluated as a parallel scan, which is why RWKV
   trains like a GPT. (v6/v7 add a *small* data-dependent component to `w` via
   a LoRA on the token-shifted input, trading a little parallelism for
   expressivity.)

### The role of token-shift / time-mixing weights

**Token-shift** (`time_shift = ZeroPad2d((0,0,1,-1))`) produces `xx = x_{t-1} -
x_t`. Every projection (`r, k, v, w, a, g`) is fed a **per-channel blend of the
current and previous token**: `x_* = x + xx · μ_*`. Effects:

* A cheap, parameter-light **1-token causal convolution** that hands each
  projection some local context *before* it hits the recurrence — RWKV's analog
  of a short receptive field.
* In **RWKV-6** the blend coefficients are themselves **data-dependent** (the
  "maa" / `time_maa_*` mechanism, `rwkv_v6_demo.py` lines 108–121): a LoRA
  produces per-token `mw, mk, mv, mr, mg` that adjust how much of the previous
  token is mixed in, per channel, per token. This is "dynamic token shift".
* In **RWKV-7** the blend is back to static per-channel coefficients
  (`x_r, x_w, x_k, x_v, x_a, x_g`, `model.py:787–792`), with the dynamic
  capacity moved into the matrix-valued state transition instead.

### The RWKV-7 advancement

RWKV-7 "Goose" is the current released generation. Its two real advances over
v6:

1. **Matrix-valued, data-dependent state transition (generalized delta rule).**
   v6's state update is `s ← diag(w)·s + (k⊗v)` (diagonal decay + write). v7's
   is `state ← diag(w)·state + state@ab + (v⊗k)` where `ab = (-kk)⊗(kk·a)`.
   The extra `state@ab` term is a low-rank, value-dependent transition that can
   **selectively erase/replace** stored associations. With `a` as an
   "in-context learning rate", the per-head state behaves like a small online
   regressor fit on the fly — strictly more expressive than exponential decay,
   and it is what closes most of the gap to attention on recall tasks (see
   `RWKV-v7/RWKV-v7-niah.png` needle-in-a-haystack results upstream).

2. **Value residual (`v_first`) — a genuine cross-layer shortcut.**
   `model.py:875–878` / `rwkv_v7_demo_rnn.py:127–130`: layer 0 stores its value
   `v_first = v`; every later layer mixes it back in,
   `v ← v + (v_first - v)·sigmoid(...)`, per channel. The first layer's "what"
   is injected, gated, into all layers. This is `v_first` threaded explicitly
   through the block stack (`Block.forward(self, x, v_first)`,
   `model.py:1077–1085`) — the one place RWKV-7 routes information *across
   layers* outside the residual stream.

**The "last layer of token N → first layer of token N+1" idea** belongs to the
**RWKV-8 roadmap**, not v7. From `RWKV-8.md` §3 "Mixed State":

> "Mixing state of the last layer of token n, with the state of the first layer
> of token n+1. A depth-L model becomes a depth-2L model after a step of this,
> and still efficiently trainable."

It is one item in a broader RWKV-8 program: the **"state ladder"** (scalar →
vector → matrix [most current RNNs] → tensor → function [where attention sits,
as kernel regression] → functional → …), plus *larger* state (bigger heads,
hybrid), *smaller* state (sparse/low-rank/shared/quantized across the
`B,T,H,N,L,Q` dimensions), *mixed* state (across heads/layers/time), and
*fancy* state evolution (`exp(sA)-1`, `1/(1-sA)`, DeltaProduct, inner
optimizers). The thesis (`RWKV-8.md` line 11) is that **attention is itself a
point on this ladder** — a "function state" doing kernel regression — and that
RNN states can climb toward it.

---

## b) Structural difference from attention-based propagation

### Where transformers carry information — the KV cache

A transformer carries history as an **explicit, growing list**: every past
token's key/value pair is stored. At step `t`, the query does a **softmax
lookup over all stored keys** and reads a convex combination of their values.

* **Memory:** `O(T)` per layer (the KV cache grows with context).
* **Compute:** `O(T)` per token, `O(T²)` per sequence.
* **Character:** *exact, content-addressable random access* to the entire
  history. Nothing is forgotten unless it falls outside the window; any past
  token can be retrieved verbatim if the query matches its key. State size is
  `f(B, T, H, N, L, Q)` — the `T` is the defining difference.

### Where RWKV carries information — the per-block evolving state

RWKV carries history as a **fixed-size state per block** that is *overwritten
in place* each token (decay + accumulate + selectively-erase). There is no
list; the past is **compressed** into the matrix `S`.

* **Memory:** `O(1)` in `T` — `f(B, H, N, L, Q)`. No KV cache.
* **Compute:** `O(1)` per token, `O(T)` per sequence (and a parallel scan for
  training).
* **Character:** *lossy, fixed-capacity, recency-and-relevance-weighted*
  memory. The model must *learn what to keep* (via decay channels + the v7
  delta-rule erase term) because capacity is bounded.

### What each is optimized for

| | Transformer / KV cache | RWKV / evolving state |
|---|---|---|
| Strength | Exact recall, content-addressable random access over the whole window; trivially parallel training | Constant memory, linear-time streaming inference; cheap "infinite" context; still parallel-trainable |
| Cost | `O(T²)` compute, `O(T)` memory; KV cache dominates long-context serving | Bounded state capacity — long-range *exact* recall must be learned into finite state |
| Optimized for | Tasks where any past token may need verbatim retrieval | Tasks where a compressed running summary suffices, and where memory/latency at long context matter |

The README's framing (lines 313–315): RWKV is "an RNN with Transformer-level
performance, which can also be directly trained like a GPT… 100%
attention-free. You only need the hidden state at position t to compute the
state at position t+1." The trade is explicit: it gives up the KV cache's exact
random access in exchange for constant-memory, linear-time inference.

---

## c) Where Aeon currently sits in this taxonomy

Read against `aeon/model.py`, `aeon/block.py`, `aeon/recursion.py`,
`aeon/config.py`. The brief's characterization is **accurate** — here it is,
confirmed against the code:

* **A pretrained transformer with a sidecar recurrent state.** `AeonModel`
  subclasses `Qwen2Model`; attention, RoPE, and the KV cache are **fully
  intact and unchanged** (`model.py:126–204`; the per-token loop threads a
  `DynamicCache` precisely so attention stays causal across the sequence). The
  recurrence is *added alongside*, it does not replace anything.

* **Recursion is an external state that modulates each block's input.**
  `AeonBlock.forward` (`block.py:69–98`): it reads the global state `r_t`,
  projects it `m = U(r_t)`, scales by a per-block gate `recursion_gate`
  (`γ_l`, **zero at init** — Stage-0 byte-identity to vanilla Qwen2), and
  **adds it to the residual stream before the Qwen block**:
  `hidden_states += (γ_l · m).unsqueeze(1)`. After the block it produces a
  write `w_l = D_proj(x_post)`. The state's only influence on computation is a
  **broadcast additive shift** — a per-token DC offset on the residual. It does
  **not** enter the attention scores, keys, values, or the MLP.

* **State advances once per token, after all blocks.** `AeonModel.forward`
  (`model.py:160–190`): for each token it runs the full `L`-block stack,
  **sums** every block's write into `W_sum`, **averages** to
  `W_total = W_sum / n_layers`, then runs **one** `recursion.step(W_total, r,
  c)`. So token `t+1` reads a state that incorporates token `t`'s contribution.
  All `L` blocks' writes are collapsed into a **single** state update per token.

* **The state object itself** (`recursion.py`, `RecursionChartB`): a
  **single, global, gate-free contractive RNN cell** with hidden `h` (= `r`)
  and a slow carry `c`, both in `R^{h_rec}` with `h_rec = 256`
  (`config.py:10`). Update:
  ```
  c_{t+1} = (1-λ)·c_t + λ·tanh(W_c h_t)         # delta-decay carry, scalar λ
  h_{t+1} = tanh(W_x x_t + W_h h_t + c_{t+1})    # contractive update
  ```
  with `W_h, W_c` built so `σ_max < margin < 1` by construction (Cayley
  transform × diagonal), i.e. a **provable contraction certificate**
  (`margin_h = 0.98`, `margin_c = 0.95`). The state is **shared across all
  layers** and persists across tokens and across chat turns
  (`get/set_recursion_state`).

* **Verdict: "transformer with sidecar recursion", not "fully integrated
  recurrent transformer".** Confirmed. The recurrence runs *parallel to*
  attention and only nudges the residual additively; attention still performs
  **all** sequence mixing.

**Aeon vs RWKV, side by side:**

| Axis | RWKV | Aeon (current) |
|---|---|---|
| State location | **Per block** (one matrix state per layer) | **One global** vector state, shared by all layers |
| State shape | `(H, N, N)` matrix per layer (high capacity) | `(h_rec,) = (256,)` vector + 256-d carry (low capacity) |
| Updates per token | `L` (one per layer) | **1** (after all blocks; writes averaged) |
| Role of state | **IS** the sequence-mixing operator (replaces attention) | **Modulates** input additively; attention still mixes the sequence |
| Coupling to compute | Integrated into the token-mix operator | **Additive DC shift** on the residual, gated by `γ_l` |
| Decay | Per-channel, per-head, learnable spectrum (+ v7 matrix erase) | Single global contraction (`σ_max<margin`) + one scalar `λ` |
| KV cache | None (`O(1)` state) | **Full Qwen KV cache** retained (`O(T)`) |
| Stability | Bounded by `w∈(0,1)` per channel | **Certified** `σ_max<1` (Banach/Lyapunov) |

On the RWKV-8 "state ladder", Aeon's recurrent state is a **vector state** (one
rung above scalar), used as a *side channel*; RWKV sits at **matrix state** and
is the main channel, with v7 reaching toward the "fancy evolution" rung.

---

## d) Open architectural questions for Aeon's next generation

Framed by what RWKV does well that Aeon currently does not. Each has a
recommendation, not just a list of options.

### 1. Per-block (RWKV) vs per-token-global (Aeon current) state?

**Finding.** Aeon collapses all `L` block writes into one 256-d update per
token (`W_total = W_sum / n_layers`). That averaging is a severe **depth
bottleneck**: every layer's contribution is summed into a single shared
register, so the recurrence cannot let layer 3's memory differ from layer 20's.
RWKV gives every layer its own state and lets each specialize a timescale.

**Recommendation: move toward per-block (or per-group) state.** Give each block
(or each band of blocks) its own recurrence cell so depth buys capacity and
layers can specialize. Keep the contractive certificate *per cell* — it
composes. Cost: `L` recurrence steps per token instead of 1 (cheap relative to
attention). A conservative first step: per-block carry, shared read/write
projections.

### 2. Integrate the state into attention, or keep it as additive modulation?

**Finding.** Additive-shift-on-residual is the **weakest possible coupling** —
the state is a per-token DC offset and never participates in *routing* (it
touches neither attention scores nor the values being mixed). RWKV's state, by
contrast, *is* the routing.

**Recommendation: a middle path — let the state participate in attention,
short of replacing it.** Concretely, inject the state as an extra **memory
key/value slot** the queries can attend to (a "register" token sourced from
`r_t`), or use it to **gate/bias the value path** (`V ← V · g(r_t)` or an
additive logit bias). This makes the state content-addressable rather than a
constant offset, while preserving the pretrained attention. Pure additive
modulation should be treated as the floor, not the design.

### 3. Learnable per-channel decay like RWKV?

**Finding.** Aeon's forgetting is a *single* global contraction plus one scalar
`λ` on the carry — effectively one (or two) timescales for the whole model.
RWKV's per-channel decay spectrum (init spanning fast→slow) is a major reason
it captures multi-timescale dependencies in a bounded state.

**Recommendation: yes, adopt per-channel decay.** Replace the single carry
`λ` with a **per-channel decay vector** (diagonal, entries in `(0,1)`),
initialized to span timescales the way RWKV does (`www`-style geometric
spread). This is **fully compatible with the stability certificate** — a
diagonal map with entries `<1` keeps `σ_max<1` — so Aeon gets multi-timescale
memory *for free* relative to its current guarantees. High value, low risk.

### 4. Replace attention entirely (full RWKV) or keep it as a parallel stream
(hybrid)?

**Finding.** Aeon is warm-started from a pretrained Qwen/R1-distill and its
Stage-0 gate demands **byte-identity to vanilla Qwen2** at init
(`block.py` gate `= 0`; `model.py` wiring notes). Full attention replacement
throws away the pretrained weights that are the entire point of the warm start.
Notably, **RWKV-8 itself is going hybrid** (the repo ships
`rwkv_v8_rc00_hybrid_demo.py`, and `RWKV-8.md` lists "hybrid models" and
"hybrid attention part" as first-class directions).

**Recommendation: hybrid, decisively.** Keep attention for exact recall and
short-range routing; grow the recurrent state for cheap, persistent, long-range
memory. This satisfies both Aeon's warm-start constraint *and* matches the
direction RWKV's own author is taking. "Full RWKV" only makes sense if Aeon
ever trains from scratch, which is not the current regime.

### Cross-cutting take

The single highest-leverage change suggested by this study is **#3
(per-channel decay)** — it is cheap, certificate-preserving, and directly
imports RWKV's multi-timescale property. The most *strategic* change is **#2
(let the state into attention)**, because Aeon's current additive coupling is
the binding constraint on how much the recurrence can ever matter. **#1** and
**#4** define the longer-term shape (more per-block capacity; stay hybrid).

---

## Appendix: key code references

| Concept | Location |
|---|---|
| RWKV block / residual stack | `RWKV-v5/rwkv_v6_demo.py:53–78`, `:457–516` |
| WKV-6 recurrence (RNN form) | `RWKV-v5/rwkv_v6_demo.py:194–221` |
| RWKV-6 time-mix (GPT form) | `RWKV-v5/src/model.py:325–418` |
| RWKV-6 channel-mix (FFN) | `RWKV-v5/rwkv_v6_demo.py:83–91`; `model.py:924–951` |
| Per-channel decay init | `rwkv_v6_demo.py:354–357`; `model.py:808–821` |
| Dynamic token-shift ("maa") | `rwkv_v6_demo.py:108–121` |
| RWKV-7 state object | `RWKV-v7/rwkv_v7_demo_rnn.py:285–288` |
| RWKV-7 delta-rule update | `rwkv_v7_demo_rnn.py:143–146`; `model.py:879–886` |
| RWKV-7 value residual (`v_first`) | `model.py:875–878`, `:1077–1085`; `rwkv_v7_demo_rnn.py:127–130` |
| WKV-7 fused kernel | `RWKV-v5/cuda/wkv7_cuda.cu:37–41` |
| Framing: attention-free / infinite ctx | `README.md:9, 313–315, 418` |
| Roadmap: state ladder, mixed state | `RWKV-8.md` §1, §3 |
| Aeon recursion cell | `aeon/recursion.py:115–170` |
| Aeon read/write per block | `aeon/block.py:69–98` |
| Aeon once-per-token state update | `aeon/model.py:160–197` |
