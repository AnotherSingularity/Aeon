# RWKV Signal-Source Study

Parallel research work: a study of [RWKV](https://github.com/BlinkDL/RWKV-LM)'s
state-propagation design as **signal-source research** for a **multi-input
contractive architecture**. In that architecture **Recursion is the substrate**
— it has its own state and σ<1 contractive dynamics, and signal sources project
into its manifold through input ports. **RWKV and the candidate recurrent
substrate (Dylan's prior work) are candidates for the RNN signal-source port**
(one input); the transformer is another source. A substrate **port spec** both
satisfy makes that choice deployment-time configuration, with Recursion as the
substrate-agnostic joiner. The design is **not limited to two sources** —
further inputs plug into Recursion's port surface without changing its substrate
nature.

> **Note on repo location.** This was meant to live in a separate repository.
> The session's GitHub access could not create or fork a new repo (the
> integration lacks repo-creation permission, and forking the upstream was out
> of session scope), so the study lives here, in the repo that was made
> available for it. The content is self-contained.

---

# Stage-1 hybrid (implementation — branch `V0.02.02`)

> ⚠️ **UNRUN.** The implementation modules were written to spec in an
> environment with no torch and no HuggingFace access. **Nothing below has been
> executed.** It is intended to be run/debugged on Vast (a rented GPU box), as
> agreed. Treat all of `aeon/recursion.py`, `aeon/transformer.py`,
> `aeon/hybrid.py`, and `scripts/train.py` as first-write-to-spec, not verified.

**No-external-architecture principle (Meaning A):** every forward-path component
is Aeon-original. `transformers` appears in **no import reachable from
`HybridModel.forward()`** — it is an optional dependency used only by the
byte-identity gate (test) and the training script's tokenizer.

The hybrid couples three sources into Recursion's σ<1 contractive manifold:

| file | role |
|---|---|
| `aeon/recursion.py` | **Recursion** — canonical two-state chart-B contractive joiner, multi-input (`W_s·s + W_t·t [+ W_e·e] + W_h·h + c`), hard `σ<margin` by Cayley construction; `step()` ticks once, `audit()` reports σ. |
| `aeon/transformer.py` | **Transformer side — Aeon-original Qwen2** (GQA+RoPE, SwiGLU, RMSNorm, pre-norm decoder, tied lm_head; **no `transformers` import**). R1 weights loaded as init via safetensors. Frozen backbone; trainable read (D→H_rec) + γ-gated write (H_rec→D, γ=0 warm start). |
| `aeon/substrate/` | **RNN signal source** behind the port (`rwkv` or `vru`, runtime-selected). |
| `aeon/hybrid.py` | **Three-source coupling** — slow-clock Recursion (K=16), running-mean window aggregation, hold-and-broadcast of the slow state to the substrate input + transformer inject, truncated BPTT at window boundaries. |
| `scripts/train.py` | YAML-driven, alpaca, bf16, batch=1, seq=512, σ/γ/loss audit logging, checkpoint + resume. |

**Config knobs added from published hybrids (both additive):**
- `model.use_embedding_input` (**default `true`**) — adds a 3rd Recursion input
  `W_e·e`, the window mean of the *original* token embeddings, giving Recursion
  direct raw-token access at integration time (Zamba-style re-injection). `W_e`
  is an input map only — the σ<margin certificate is unaffected.
- `model.substrate.use_state_norm` (**default `false`**) — optional RMSNorm on
  the RWKV-class cell's per-head state before the receptance read, to control
  matrix-state magnitude drift at scale (cf. Jamba). Read-path only; the stored
  accumulator keeps its raw dynamics. A debug knob if Vast shows S drift.

## Install (CUDA 12.4)

```bash
# torch must come from the cu124 index (pinned 2.5.1)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e .            # core architecture: torch + safetensors + pyyaml (NO transformers)
pip install -e ".[train]"   # + transformers (tokenizer), datasets, huggingface_hub, accelerate
pip install -e ".[test]"    # + transformers + pytest (for the byte-identity gate)
```

`transformers` is an optional extra, never imported by the `aeon/` architecture.

## Byte-identity gate (run FIRST — load-bearing proof before any training)

The R1 warm-start is only trustworthy if Aeon's transformer reproduces HF Qwen2
exactly. Download R1 locally, then:

```bash
pip install -e ".[test]"
AEON_R1_DIR=/path/to/DeepSeek-R1-Distill-Qwen-1.5B python tests/test_byte_identity.py
```

Pass criteria: **bf16 bit-identical** (the warm-start dtype — verified max|Δ| =
0.0 on a V100 once RoPE was computed in fp32); **fp32 within 1e-3** (~5.5e-4 is
the eager-kernel reduction-order noise floor, not a bug — fp32 is not a training
dtype). The bf16-at-γ=0 guarantee is the load-bearing one and it holds exactly.

## Running Stage-1 hybrid on Vast

```bash
python scripts/train.py --config configs/stage1_hybrid.yaml
```

The first run downloads the R1-Distill-Qwen-1.5B checkpoint and the alpaca
dataset (needs HF egress — blocked in the authoring sandbox, available on Vast).
Training is resumable: re-running picks up the latest `runs/stage1_hybrid/ckpt_*.pt`.

## Expected outputs (what a healthy run looks like)

- `[init] audit @ start: {... 'holds': True, 'gamma': 0.0}` — at init γ=0, so the
  hybrid is byte-identical to plain R1 (warm start); the σ certificate holds.
- Per-step audit lines:
  `[step N] loss=… sigma_Wh=… sigma_Wc=… holds=True lambda=… gamma=…`
  — `holds` must stay `True` every step (the certificate is structural; a
  `False` is a bug, flagged with `[WARN]`). `gamma` should grow away from 0 as
  the recurrent signal starts to matter.
- Checkpoints every `ckpt_every` steps in `runs/stage1_hybrid/`.

## Verification to do on a GPU/CPU box with the checkpoint (deferred per plan)

0. **Byte-identity gate (FIRST, load-bearing):** Aeon transformer == HF Qwen2 on
   identical weights+inputs (`tests/test_byte_identity.py`). Gate everything else
   on this.
1. R1 checkpoint loads cleanly into the Aeon backbone (`load_pretrained`).
2. `model(input_ids, labels=…)` forward runs; loss computes; `loss.backward()`;
   `opt.step()` — one end-to-end step.
3. γ=0 warm-start check: hybrid logits == `transformer.plain_logits(...)`.
4. σ certificate holds across steps.

## Open design decisions flagged in-code (confirm against HYBRID_DESIGN.md)

See the module docstrings (`hybrid.py` D1–D4, `recursion.py` interpretations
(1)–(2)) for the design choices I **derived** from the relayed answers rather
than received verbatim — chiefly: the held conditioning state is the *previous*
window's tick output (causal, no leakage), and truncated BPTT flows gradient one
window back. Confirm before a long training run.

---

## Contents

- **[`docs/RWKV_STUDY.md`](docs/RWKV_STUDY.md)** — the analysis. Covers how
  state propagates in RWKV (per-block matrix state, time-mix recurrence,
  per-channel decay, token-shift, the RWKV-7 delta-rule + value-residual), the
  structural contrast with attention/KV-cache, RWKV read as a **candidate RNN
  signal source** (the read/write *ports* it presents to Recursion), a
  **substrate port spec** that both the candidate recurrent substrate (Dylan's
  prior work) and RWKV-class blocks satisfy — so substrate choice is
  deployment-time configuration, not an architectural commitment, with Recursion
  as the substrate-agnostic joiner — an **argued position** on the port-spec
  design (minimal-common vs required+optional capability tiers), and **positions
  across** the multi-input substrate design space — including that the
  architecture is not structurally limited to two sources. Positions are input
  to deliberation, not decisions.
- **[`reference/PROVENANCE.md`](reference/PROVENANCE.md)** — audit record. A
  read-only subset of RWKV-LM was briefly vendored for the study and has been
  **removed under the no-external-codebases principle**; the study's citations
  now link to upstream `BlinkDL/RWKV-LM` at a pinned commit (Apache-2.0).

## Reading order

1. `docs/RWKV_STUDY.md` — start here.
2. Follow its citations out to upstream `BlinkDL/RWKV-LM` (pinned commit; see the
   doc's Appendix).
