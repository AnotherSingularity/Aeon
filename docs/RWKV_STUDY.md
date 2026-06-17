# RWKV as a Recurrent Substrate — State-Propagation Study

Substrate research for a **three-ingredient hybrid**:

1. a recurrent **substrate** — carries an evolving state (candidate fillers:
   **VRU**, or **RWKV** studied fresh here; *the slot is open*);
2. a transformer **reasoner** — attention-based reasoning;
3. **Recursion** — the **application layer that couples** the substrate to the
   reasoner. Not the substrate, not a modulator on either side: it *owns the
   coupling*.

This document studies **RWKV purely as a candidate for ingredient #1** — how a
real, scaled recurrent architecture propagates state, and what *ports* it would
expose to a coupling layer. It deliberately does **not** decide VRU-vs-RWKV
(§d), nor how the three layers couple (§e); those stay open for the architect.
Parts (a)–(b) are the substrate mechanics; (c) re-reads them as an *interface*;
(d) frames the substrate decision; (e) maps the coupling design space.

**Source studied:** [`BlinkDL/RWKV-LM`](https://github.com/BlinkDL/RWKV-LM).
A focused, text-only subset of the studied files is vendored under
[`../reference/RWKV-LM/`](../reference/RWKV-LM/) so the file/line citations
below resolve in-repo (the full upstream tree carries ~5 MB of images plus many
model generations that the study did not need — see that folder's
`PROVENANCE.md`).

**Aeon source referenced** as one concrete data point on a VRU-class cell — not
as the target architecture: the `aeon/` package (`model.py`, `block.py`,
`recursion.py`, `config.py`).

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

## c) RWKV as a candidate recurrent substrate — its state interface ("ports")

A substrate is judged on two things: how well it *holds* information (parts a–b)
and what **ports** it exposes for a coupling layer to read from and write to.
Read off the RNN-form code, RWKV's ports are unusually rich:

**Read ports — what a reasoner could consume:**

* **State readout** `out = state @ r` (v7) — the canned "what the substrate
  knows now", `C`-dim per layer; already the time-mix output.
* **Raw matrix state** `S` of shape `(H, N, N)` per layer — the full
  associative memory. A coupling layer can expose `S` itself (or a *learned
  query* of it) instead of only the built-in readout `r`. This is the port that
  makes a true cross-attention-style read possible.
* **Per-layer or pooled** — RWKV keeps one `S` per layer, so the coupling layer
  can read at any depth, or pool across depth.

**Write ports — what a reasoner could push into the substrate:**

* **Association write** `S += v ⊗ k` — inject a key→value memory directly; a
  reasoner can write conclusions as retrievable associations.
* **Decay control** `w` (per channel, `∈(0,1)`) — set how long things persist;
  modulate memory horizon per channel.
* **Delta-rule erase / learning-rate** `a`, `kk` (v7) — drive in-context
  write/overwrite, so the substrate is *programmable while reading*.
* **Input mixing** — the standard token-shifted `x` input is the cheapest write.

**Cross-layer bus.** v7's `v_first` (`model.py:875–878`) shows RWKV already
supports a value bus threading layer 0 → all layers — a coupling layer could
ride or extend it.

**Clocking.** The substrate steps **`L` times per token** (once per layer,
inside the operator), so coupling can be per-layer or batched at a readout.

**Net:** RWKV offers a coupling layer a **high-capacity, content-addressable,
multi-timescale, writable** memory with linear-time / constant-space dynamics
and a parallel-scan training path. That is a *rich* substrate — many more ports
than a single-vector cell.

**Contrast: a VRU-class cell.** Using the certified contractive cell in
`aeon/recursion.py` as a concrete stand-in (flag: swap in the real VRU spec when
you fix it), the port surface is far smaller — one input `x_t` (write), one
hidden `h_t` (read), a slow carry `c`, and a single global contraction + scalar
`λ` for decay. Its strength is **guarantees, not interface richness**:
`σ_max < margin < 1` (`margin_h=0.98`, `margin_c=0.95`) gives provable
forgetting, a unique attractor per fixed input, and bounded sensitivity —
properties a coupling layer can *rely on* rather than police. Two substrate
philosophies: **RWKV = capacity & rich ports; VRU-class = bounded, certified,
simple ports.**

> **Where the *current* Aeon code sits is now just a data point, not the
> target.** It fuses a transformer with a VRU-class cell used as an **additive
> modulator**: `AeonBlock` reads the global state, gates it by `γ_l` (zero at
> init), and adds it to the residual before the Qwen block (`block.py:69–98`);
> the state advances **once per token** after the full stack, on the *average*
> of all block writes (`model.py:160–197`). That is the **one-sided** design
> being moved past — substrate and coupling are not factored apart, and the
> recurrence only nudges the residual. The three-ingredient frame separates
> them; this study treats the substrate slot as genuinely open.

---

## d) The substrate decision — RWKV vs VRU (kept open)

Decision deferred to the architect. These are the axes any recurrent substrate
should be judged on *for this role*, with RWKV filled from the study and the VRU
column filled hypothetically from `recursion.py` (flagged — replace with the
real spec):

| Axis | RWKV (studied) | VRU-class (per `recursion.py`; hypothesis) |
|---|---|---|
| State shape / capacity | `(H,N,N)` matrix per layer — **high** | `(h_rec,)` vector + carry — **low** |
| State-evolution expressivity | v7 data-dependent matrix transition (delta rule) | contractive affine + `tanh` |
| Decay / multi-timescale | per-channel learnable spectrum — **strong** | single contraction + scalar `λ` — weak |
| Stability guarantee | bounded `w∈(0,1)`; no global cert | **certified** `σ_max<1` (Banach/Lyapunov) |
| Trainability / parallelism | **parallel scan** ("GPT mode"), proven at scale | sequential BPTT (nonlinear recurrence) |
| Coupling ports (read/write) | **rich** (matrix `S`, decay, delta-rule, bus) | sparse (one in, one out) |
| Constant-space inference | yes (`O(1)` in `T`) | yes |
| Maturity / warm-start | pretrained checkpoints exist | bespoke |
| Controllability / interpretability | lower | **high** (provable bounds) |

The real tension: **capacity + rich ports + maturity (RWKV)** versus
**certified control + simplicity (VRU)**. Which wins depends on what the
substrate is *for* in the coupling (§e): if Recursion leans on provable
substrate behavior, VRU's guarantees are load-bearing; if it leans on the
substrate as a big queryable/writable memory, RWKV's ports and capacity win. So
§d and §e are entangled but not identical decisions.

**To slot VRU in precisely I'd need:** its state shape/capacity; whether its
recurrence is linear (parallel-scannable) or nonlinear (sequential); and which
read/write ports you intend to expose to the coupling layer.

---

## e) The coupling question — how Recursion joins substrate ↔ reasoner

This is the architecture question you're sitting with; I'm **mapping the design
space, not choosing**. Recursion-as-application-layer has (at least) five
roughly orthogonal degrees of freedom. The substrate choice (§d) constrains
some of them, not all.

**A. Read coupling (substrate → reasoner).** How does the reasoner *see* state?
Floor → rich: additive residual bias (current Aeon) → state as extra **KV
memory slot(s)** the reasoner attends to (content-addressable) → state
**gates** attention values / FFN → reasoner **cross-attends into `S`**.
*Substrate-dependent:* RWKV's `(H,N,N)` `S` supports a true cross-attention
read; a vector VRU supports bias/gating/KV-slot but not a rich matrix query.

**B. Write coupling (reasoner → substrate).** What updates the substrate?
Reasoner hidden states projected as substrate input (cheap) → reasoner **writes
associations into `S`** (RWKV-only) → reasoner **controls decay / learning-rate
knobs**.

**C. Clocking / schedule.** When does the substrate step relative to the
reasoner? Once per token after the full stack (current Aeon) → per-layer
(RWKV-native) → substrate on a **slower or faster clock** (the substrate
"thinks" across multiple reasoner passes, or persists across turns while the
reasoner restarts). This is where genuine three-way timing lives.

**D. Loop topology.** Feedforward (substrate informs reasoner only) vs **closed
loop** (reasoner updates substrate → informs the next step). "Application
layer" implies Recursion owns a closed loop; the open question is how tight.

**E. Where Recursion's intelligence lives.** Thin interface (fixed projections)
vs **substantive controller** with its own parameters/policy deciding *what* to
read/write and *when*. Your framing — "application layer that couples" — points
substantive: Recursion as a learned controller over the substrate's ports, not
glue.

Two observations to carry in:

* **A/B (coupling richness) and §d (substrate) are coupled.** Rich coupling
  needs a substrate with rich ports — so "RWKV vs VRU" and "how rich is the
  coupling" want to be decided *together*, not in sequence.
* **C/D (clocking, loop topology) are largely substrate-independent.** You can
  fix the three-way timing and whether Recursion runs a closed loop *before*
  committing the substrate. Those may be the cleanest places to start.

Per your division of labor I stop at the map — no coupling recommendation. The
substrate research is in hand for the VRU-vs-RWKV call.

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
