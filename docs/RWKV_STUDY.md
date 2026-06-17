# RWKV as an RNN Signal Source — State-Propagation Study

Signal-source research for a **multi-input contractive architecture**.

The architecture: **Recursion is the substrate.** It has its own state and its
own σ<1 contractive dynamics — *that contractive evolution is the manifold.*
Multiple **signal sources project into it** through input ports; Recursion
integrates them and evolves. The current instance has two sources:

1. an **RNN signal source** (candidate fillers: **VRU** or **RWKV**, studied
   fresh here);
2. a **transformer signal source** (attention-based);

…feeding **Recursion**, the multi-input contractive substrate they project into.
This is **not** structurally a two-source design: additional sources — other
modalities, error signals, specialized processors — plug into Recursion's port
surface without changing its substrate nature. "RNN + transformer + Recursion"
is the *current instance*, not the structural limit.

This document studies **RWKV purely as a candidate for the RNN signal-source
slot** — one of Recursion's input ports — and what it exposes for Recursion to
ingest. It does **not** decide RWKV-vs-VRU for that port (§d), nor fix the
multi-input design (§e); those are the architect's calls. Parts (a)–(b) are
RWKV's state-propagation mechanics; (c) re-reads them as the *ports* RWKV would
present to Recursion; (d) frames the RNN-source decision; (e) maps the
multi-input substrate design space.

**Source studied:** [`BlinkDL/RWKV-LM`](https://github.com/BlinkDL/RWKV-LM).
A focused, text-only subset of the studied files is vendored under
[`../reference/RWKV-LM/`](../reference/RWKV-LM/) so the file/line citations
below resolve in-repo (the full upstream tree carries ~5 MB of images plus many
model generations that the study did not need — see that folder's
`PROVENANCE.md`).

**Aeon source referenced:** the `aeon/` package — `recursion.py` is the
**Recursion substrate** (the `σ_max<margin<1` Cayley cell), and
`block.py`/`model.py` show the *current* wiring (a one-sided instance being
moved past, see §c).

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

## c) RWKV as a candidate RNN signal source — the ports it presents to Recursion

RWKV would occupy **one of Recursion's input ports** (the RNN signal source).
What matters for that role is not only how RWKV holds state (parts a–b) but the
**ports it presents**: what Recursion can ingest from it, and what Recursion (or
a feedback loop) can push back into it. Read off the RNN-form code, RWKV's port
surface is unusually rich.

**Read ports — what Recursion can ingest from RWKV:**

* **State readout** `out = state @ r` (v7) — the canned "what the source knows
  now", `C`-dim per layer; already the time-mix output.
* **Raw matrix state** `S` of shape `(H, N, N)` per layer — the full
  associative memory. Recursion can ingest `S` itself (or a *learned query* of
  it) instead of only the built-in readout `r`. This is the port that makes a
  true cross-attention-style read possible.
* **Per-layer or pooled** — RWKV keeps one `S` per layer, so Recursion can read
  at any depth, or pool across depth.

**Write ports — what Recursion (or a feedback loop) can push into RWKV:**

* **Association write** `S += v ⊗ k` — inject a key→value memory directly.
* **Decay control** `w` (per channel, `∈(0,1)`) — set how long things persist;
  modulate the source's memory horizon per channel.
* **Delta-rule erase / learning-rate** `a`, `kk` (v7) — drive in-context
  write/overwrite, so the source is *programmable while reading*.
* **Input mixing** — the standard token-shifted `x` input is the cheapest write.

**Cross-layer bus.** v7's `v_first` (`model.py:875–878`) shows RWKV already
carries a value bus threading layer 0 → all layers — Recursion could ride or
extend it.

**Clocking.** RWKV's own state steps **`L` times per token** (once per layer,
inside the operator). That sets the source's *native* update cadence, which any
sampling against it (§e) has to reckon with.

**Net:** as an RNN signal source, RWKV presents Recursion a **high-capacity,
content-addressable, multi-timescale, writable** port surface, with
linear-time / constant-space dynamics and a parallel-scan training path — many
more ports than a single-vector source.

**Contrast: a VRU-class source.** VRU's spec is open; as a *character*
placeholder for "a certified, simple RNN source" (real spec TBD), such a cell
presents far fewer ports — one input (write), one hidden read, perhaps a decay
knob. Its strength is **guarantees, not port richness**: a `σ_max<1` contractive
cell gives provable forgetting, a unique attractor per fixed input, and bounded
sensitivity — properties Recursion can *rely on* rather than police. Two source
philosophies: **RWKV = capacity & rich ports; VRU-class = bounded, certified,
simple ports.**

> **The current Aeon code is a *one-sided* instance of this architecture — the
> thing being moved past.** Today only the **transformer** projects into
> Recursion: each block emits a write, the writes are **averaged** into a single
> per-token step (`model.py:160–197`), and Recursion's state is read back only
> as a **gated additive shift** on the residual (`block.py:69–98`, `γ_l=0` at
> init). There is **no RNN signal source**, and the port surface is a single
> averaged write in / a DC offset out. Adding RWKV (or VRU) as a second input
> port — and widening the ports — is exactly the correction.
> `recursion.py`'s `RecursionChartB` is the **Recursion substrate** itself (the
> `σ_max<margin<1` Cayley cell, `margin_h=0.98`, `margin_c=0.95`) — not a signal
> source.

---

## d) The RNN-source decision — RWKV vs VRU (kept open)

Which candidate fills Recursion's **RNN signal-source port**. Decision deferred
to the architect. These are the axes for judging an RNN-source candidate, with
RWKV filled from the study and the VRU column a flagged hypothesis (replace with
the real spec):

| Axis | RWKV (studied) | VRU-class (certified-contractive hypothesis; spec TBD) |
|---|---|---|
| State shape / capacity | `(H,N,N)` matrix per layer — **high** | vector + carry — **low** |
| State-evolution expressivity | v7 data-dependent matrix transition (delta rule) | contractive affine + `tanh` |
| Decay / multi-timescale | per-channel learnable spectrum — **strong** | single contraction + scalar `λ` — weak |
| Stability guarantee | bounded `w∈(0,1)`; no global cert | **certified** `σ_max<1` (Banach/Lyapunov) |
| Trainability / parallelism | **parallel scan** ("GPT mode"), proven at scale | sequential BPTT (nonlinear recurrence) |
| Port surface to Recursion | **rich** (matrix `S`, decay, delta-rule, bus) | sparse (one in, one out) |
| Constant-space inference | yes (`O(1)` in `T`) | yes |
| Maturity / warm-start | pretrained checkpoints exist | bespoke |
| Controllability / interpretability | lower | **high** (provable bounds) |

The real tension: **capacity + rich ports + maturity (RWKV)** versus
**certified control + simplicity (VRU)**. Which wins depends on what Recursion
needs *from* the RNN port (§e): if Recursion leans on provable source behavior,
VRU's guarantees are load-bearing; if it leans on the source as a big
queryable/writable memory, RWKV's ports and capacity win. So §d and §e are
entangled but not identical decisions.

**To slot VRU in precisely I'd need:** its state shape/capacity; whether its
recurrence is linear (parallel-scannable) or nonlinear (sequential); and which
read/write ports it exposes to Recursion.

---

## e) Multi-input substrate design space

Recursion is a **multi-input contractive substrate**: signal sources project
into its σ<1 manifold through input ports, and it integrates them and evolves.
The design space is therefore **per-source port design** layered over the
substrate's own dynamics. I'm **mapping it, not choosing**. Five roughly
orthogonal degrees of freedom, stated per-source:

**A. Per-source read port shape** — what Recursion ingests *from* a source.
Floor → rich: additive bias → the source's state as **KV memory slot(s)**
Recursion attends to → the source **gates** internal computation → Recursion
**cross-attends into the source's state**. *Source-dependent:* RWKV's `(H,N,N)`
`S` supports a true cross-attention read; a vector source supports
bias/gating/KV-slot but not a rich matrix query.

**B. Per-source write port shape** — what Recursion pushes *back to* a source.
None (feedforward source) → Recursion's output projected as the source's input →
Recursion **writes directly into the source's state / controls its knobs**
(RWKV's `S`, decay, learning-rate).

**C. Per-source sampling rate** against Recursion's contractive clock. Each
source can be sampled at its own cadence relative to the substrate's evolution —
fast sources every step, slow ones sub-sampled, or the substrate run faster than
any source.

**D. Loop topology — per source.** Whether Recursion's output feeds **back** to
each source (closed loop) or the source is read-only into the substrate
(feedforward). Can differ source to source.

**E. Where the cognitive work lives.** The distribution between the specialized
sources (RWKV's associative memory, the transformer's attention) and Recursion's
**integrating** substrate — how much each source pre-computes versus how much
integration the contractive manifold does.

Two observations to carry in:

* **Per-source read-port richness (A) and the source choice (§d) are entangled.**
  The richness of Recursion's RNN-input port is **capped by what the RNN source
  exposes**: RWKV's matrix `S` admits a true cross-attention read; a vector VRU
  admits bias/gating/KV-slot. Source choice and port shape want to be decided
  *together*.
* **Sampling rate (C) and loop topology (D) are source-independent at the
  framing level, but implementation cost is not.** RWKV-7's matrix state updates
  **per token by construction**, so sampling it slower means bypassing updates
  or holding state; a VRU's cadence is configurable. The genuinely novel
  property generalizes beyond two sources: **slow-clock contractive integration
  across multiple fast-clock signal sources** is something no single source —
  RNN or transformer — has alone.

### Architectural property: not limited to two sources

**Load-bearing framing, not a detail.** The architecture is not structurally
"RNN + transformer + Recursion". It is **Recursion as a multi-input contractive
substrate** that *currently* has two sources but is not limited to two. The σ<1
manifold and its port surface are the invariant; additional sources — other
modalities, error signals, specialized processors — plug into new input ports
without changing Recursion's substrate nature. "RNN + transformer" is the
present instance, not the ceiling. Every axis above (A–E) is defined *per source*
precisely so the frame holds as ports are added.

Per the division of labor I stop at the map — no source or design-space
recommendation. The RWKV signal-source research is in hand for the RWKV-vs-VRU
call.

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
