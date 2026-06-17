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
ingest. Parts (a)–(b) are RWKV's state-propagation mechanics; (c) re-reads them
as the *ports* RWKV would present to Recursion; (d) **argues a position** on the
RNN-source decision; (e) **takes positions** across the multi-input substrate
design space. The positions in (d)–(e) are **input to the architect's
deliberation, not decisions**.

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

## d) The RNN-source decision — RWKV-7 vs the contractive recurrent substrate

Two candidates for Recursion's **RNN signal-source port**: **RWKV-7**, and the
**contractive recurrent substrate** that is Dylan's prior work (a σ<1 cell; its
concrete proxy here is the Cayley cell in `recursion.py`). This section **takes
a position** — argues a choice, surfaces the trade-offs per axis. The position
is **input to deliberation, not a decision**; the call is Dylan's.

> **Information-asymmetry flag — read before weighting anything below.** The two
> columns are *not* equally evidenced. RWKV-7's ports and dynamics are read from
> deep source (multiple versions, the fused CUDA kernels, RNN- and GPT-form
> demos) → **high confidence**. The contractive substrate is known only from a
> **brief port spec at the disclosure boundary** (the `recursion.py` cell plus
> what's been said) → **lower confidence**. Treat the substrate-side cells below
> as provisional. If the real prior work is richer than the brief spec — e.g. a
> matrix state, or more ports than `recursion.py` shows — the *port-richness*,
> *parameter-efficiency* and *long-context* axes move toward parity and the
> position must be revisited. The deliberation should correct for this
> asymmetry rather than inherit a false symmetry between the columns.

| Axis | RWKV-7 (deep-source; high conf.) | Contractive substrate (brief spec; low conf.) |
|---|---|---|
| Port richness | **rich** — raw matrix `S` read, association writes, per-channel decay & learning-rate knobs, `v_first` bus | **sparse** — one input projection in, one vector read out, scalar `λ` knob |
| Recurrence type | linear-in-state WKV + data-dependent diagonal & low-rank (delta-rule) transition | **nonlinear** (`tanh`) contractive map + slow carry |
| Parallel-scannability | **yes** — associative scan / chunked kernel ("GPT mode") | **no** — nonlinear recurrence ⇒ sequential BPTT over `T` |
| Stability mechanism | architectural: per-channel `w∈(0,1)`; **no global certificate** | **certified** `σ_max<1` (Banach/Lyapunov): provable forgetting, bounded sensitivity |
| Parameter efficiency | **heavy** — effectively a second full sequence model (`C×C` projections/layer + matrix state) | **light** — a couple of `H×H` matrices + input projection |
| Long-context behavior | **strong** — constant memory, multi-timescale decay, delta-rule recall (NIAH) | **weak** — low-capacity vector state + contraction ⇒ aggressive forgetting, poor exact long-range recall |
| Maturity | pretrained checkpoints, kernels, scaled | bespoke, unproven at scale, theoretically clean |

**Position: RWKV-7 for the RNN input port — provisionally, and for one specific
reason.** The RNN port exists to give Recursion what the attention-based source
*cannot* hold cheaply: high-capacity, queryable, persistent memory in constant
space. On every axis that serves that job — port richness (a matrix `S`
Recursion can cross-attend into), long-context retention, parallel-scannable
training, maturity — RWKV-7 dominates. The contractive substrate's one decisive
win is the **certified σ<1**, and that is exactly the property **Recursion
already owns**: its global manifold is contractive by construction. Duplicating
the certificate inside the source buys little; *not* duplicating it frees the
source to be high-capacity rather than contraction-limited. The cleaner division
is therefore **certified stability lives in Recursion; capacity lives in the RNN
source (RWKV-7); the contractive cell stays where it belongs — as Recursion's
own substrate dynamics, not as the source.**

**What would flip this** (the trade-offs that make it a position, not a verdict):

* **Parameter budget.** RWKV-7 as the RNN source means running a second full
  sequence model alongside the Qwen2-family backbone. If that cost is
  unacceptable, the light contractive cell wins on efficiency and the design
  must lean harder on the transformer side for capacity.
* **Global-certificate scope.** If the architecture requires the *whole* system
  (sources included) to carry the σ<1 guarantee — not just Recursion's manifold
  — then an uncertified RWKV source must be wrapped/gated to preserve
  contraction, eroding its advantage, and a natively certified contractive
  source becomes the safer fill.
* **Asymmetry resolving.** If the brief substrate spec understates the prior
  work (richer ports / a matrix state), the gap narrows and the
  efficiency + certified-stability combination could carry it.

So: **RWKV-7, conditional on (a) accepting a second sequence model's parameter
cost and (b) the σ<1 guarantee being Recursion-local.** Both conditions are
Dylan's to confirm; the position is input to that deliberation.

---

## e) Multi-input substrate design space — positions per source

Recursion is a **multi-input contractive substrate**: sources project into its
σ<1 manifold, it integrates and evolves. The design space is **per-source port
design** over the substrate's own dynamics. Same stance as §d — **a position or
tentative direction per axis, with trade-offs**, not questions handed back. All
of it **conditions on §d's argued choice (RWKV-7 in the RNN port)**; each axis
closes with how it shifts **if §d goes the other way** (the contractive cell as
the source). The two sources in the current instance are the **RNN source** and
the **transformer side** (Qwen2-family backbone).

**A. Per-source read port shape** — what Recursion ingests.
*Position.* RNN source: **cross-attend into RWKV's matrix `S`** (a learned query
over per-layer-pooled `S`), not the canned vector readout — the matrix is the
whole reason to pick RWKV, and collapsing it to a vector wastes the port.
Transformer side: read **residual-stream hidden states** (final layer plus a few
mid-depth taps), projected into the manifold; the transformer's "state" *is* its
activations, so this is the natural, cheap read. *Trade-off.* Cross-attending
`(H,N,N)` per layer is costly — mitigate with layer-pooling and a low-rank query.
*If §d flips:* the RNN read collapses to a vector (bias / gate / KV-slot);
Recursion gets less from the RNN source and must lean on the transformer taps for
capacity.

**B. Per-source write port shape** — what Recursion pushes back.
*Position.* Keep the RNN source **feedforward (no write-back) in v1** to bound
complexity; when a loop is wanted, write through the **low-dimensional knobs**
(decay `w` / learning-rate `a`), *not* direct `S` association writes — knob
control is stable and cheap, direct `S` writes risk destabilizing the source and
are hard to train. Transformer side: write-back is **injection into the residual
stream** — the current gated additive shift is the floor; prefer a **gated KV
memory slot** the attention can attend to, kept `γ`-gated from 0 to protect the
warm-start. *Trade-off.* Richer write-back = more expressivity and more ways to
break a pretrained backbone. *If §d flips:* the contractive source has no rich
knobs, so its write port is just its input projection — low-bandwidth, but safe
to close the loop on immediately (it's certified).

**C. Per-source sampling rate vs Recursion's contractive clock.**
*Position.* Run **Recursion on a slower clock than both sources** — sources
sampled per token, Recursion integrating every `k` tokens (or per segment). A
slow contractive clock doing multi-timescale integration over fast sources is
the architecture's distinctive move (finding 2). *Trade-off.* A slower Recursion
clock makes its influence staler within a window — tune `k`: too slow and
continuity lags, too fast and there's no integration benefit. *RWKV specifics:*
RWKV-7's `S` updates **per token by construction**, so "sample slower" means
**aggregating/holding its readout** across the window, not retiming its
internals. *If §d flips:* the contractive cell's cadence is freely configurable,
so C becomes a free parameter rather than an aggregation problem.

**D. Loop topology — per source.**
*Position.* v1: **transformer side = closed loop** (Recursion → residual is the
whole point — continuity feeding back into generation); **RNN source =
feedforward** into Recursion, fed the same token stream as a parallel source.
Closing the loop into RWKV's state is the riskiest connection to build; defer it
until knob-control write-back (B) is proven. *Trade-off.* A feedforward RNN source is
not shaped by Recursion — acceptable, because it still sees the same input and
carries its own long memory. *If §d flips:* closing the loop on a certified
contractive source is safe from v1, so both sources could be closed-loop
immediately.

**E. Where the cognitive work lives.**
*Position.* **Distribute, don't centralize.** Transformer side: exact
within-window association (parallel attention). RNN source: long-horizon,
constant-memory persistence — bias RWKV toward its **slow decay channels** so it
specializes in what attention cannot hold rather than duplicating short-range
work. Recursion: **cross-source integration and continuity** — the bounded,
slow-clock manifold that fuses the sources and carries identity across turns. No
single layer does "the reasoning"; **reasoning is the emergent behavior of the
integrated system.** *Trade-off.* A clean split risks redundancy (both attention
and RWKV doing within-window association) — the decay-channel bias is the lever
that prevents it. *If §d flips:* a low-capacity contractive source cannot carry
long memory, so persistence falls back onto Recursion's own (low-capacity)
manifold or the transformer's KV cache — the latter partially defeats the
constant-memory aim. This is the strongest *system-level* argument for RWKV in §d.

### Architectural property: not limited to two sources

**Load-bearing framing, not a detail.** The architecture is not structurally
"RNN + transformer + Recursion". It is **Recursion as a multi-input contractive
substrate** that *currently* has two sources and is not limited to two. The σ<1
manifold and its port surface are the invariant; further sources — other
modalities, error signals, specialized processors — plug into new ports without
changing Recursion's nature. Every axis above (A–E) is stated *per source*
precisely so the positions extend to a third or fourth source unchanged.

These are positions to deliberate against, not decisions — and per §d's
asymmetry flag, any position resting on the substrate-side spec is held at lower
confidence than the RWKV-side analysis.

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
