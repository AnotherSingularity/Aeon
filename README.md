# Aeon

A multi-source contractive architecture. Aeon couples a recurrent **substrate**
and a **transformer side** through **Recursion** — a multi-input contractive
joiner whose state lives on a σ<1 manifold (a hard certificate, by construction).
Every signal source projects into that manifold; Recursion integrates them on a
slow clock and conditions generation.

**100% Aeon-original, weights and code.** Random-initialized, trained end-to-end.
No external architecture and no external library in any forward path.

## Architecture

| component | file | role |
|---|---|---|
| **Recursion** | `aeon/recursion.py` | the σ<1 contractive joiner. Two-state cell (`h` + delta-decay carry `c`); recurrent weights `W = sigmoid(s)·MARGIN·Cayley(A)·diag(tanh(d))` give `σ < MARGIN` by construction. Multi-input: `h = tanh(W_s·s + W_t·t + W_e·e + W_h·h + c)`. `audit()` reports σ. |
| **Substrate** | `aeon/substrate/` | the recurrent signal source behind a read/write/cadence **port**. Two cells: `matrix_cell` (matrix state, per-channel decay, outer-product write) and `vector_cell` (single-vector state, deliberately simple). Cell choice is deployment-time config via `make_substrate()`. `verify_substrate()` is the conformance gate. |
| **Adaptive feedback** | `aeon/substrate/feedback.py` | closed-loop load control on the `matrix_cell` readout: a load sensor `L(t)`, a smooth gate `g(L)=σ(α(L−θ))`, and a bound-preserving stressed-mode blend. Open-loop at low load; under stress it sharpens the substrate's output direction so Recursion drives a correction. Certificate holds in every mode (see below). |
| **Transformer side** | `aeon/transformer.py` | Aeon's own transformer: RMSNorm, rotary embeddings, grouped-query attention, SwiGLU MLP, pre-norm decoder stack, tied head. Read surface (hidden→manifold) and γ-gated write surface (manifold→hidden). |
| **Coupling** | `aeon/hybrid.py` | slow-clock Recursion (`K`-token windows): running-mean window aggregation, hold-and-broadcast of the previous window's state to the substrate input and the transformer inject, truncated BPTT at window boundaries. |

## Install (CUDA 12.4)

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e .            # safetensors, pyyaml, numpy
```

## Model scale

`configs/aeon_350m.yaml` is the from-scratch prototype target: **~350.28M
trainable** (hidden 1024, 24 layers, 16 heads × 64, GQA with 4 KV heads,
intermediate 2048, substrate/manifold `h_rec`=512, slow clock K=16, **128k
multilingual vocab**, adaptive feedback on). The 128k×1024 tied embedding is
131.07M; everything-else-transformer is 218.94M; adaptive feedback adds 0.26M
(`W_stressed` + gate scalars). `configs/aeon_v1.yaml` is a smaller smoke config.
Everything is random-initialized.

## Tokenizer (Aeon's own, from scratch)

Aeon trains its **own** 128k **multilingual** tokenizer (top-50 languages) —
nothing adopted, nothing downloaded. The backend is SentencePiece with UTF-8
byte-fallback (so any CJK/Indic/Arabic code point decomposes to bytes rather than
`<unk>`); the produced `.model` is versioned alongside the weights.

```bash
python scripts/train_tokenizer.py --corpus data/aeon_corpus \
    --out tokenizer --name aeon --vocab-size 128000
```

The corpus is a `.txt` file (one record per line), a `.jsonl` file (records under
a `"text"` field), or a directory of them (`aeon/data.py` is the one reader, used
by both training and tokenizer training). Special-id layout is fixed —
`pad=0, unk=1, bos=2, eos=3` — so retrains stay checkpoint-compatible.

## Run

```bash
# real run: point the config's data.tokenizer + data.corpus at your artifacts
python scripts/train.py --config configs/aeon_350m.yaml
```

Set `data.tokenizer` (an Aeon `.model`) and `data.corpus` (text/jsonl/dir) in the
config to train on a real tokenized corpus (single-epoch packed batches); leave
both `null` and `train.py` falls back to a **synthetic random-token** source so the
pipeline still runs end-to-end (forward / loss / backward / optimizer step /
certificate audit / checkpoint, resumable).

Healthy audit lines look like:
`[step N] loss=… sigma_Wh=… sigma_Wc=… holds=True lambda=… gamma=…` — `holds`
stays `True` every step (the certificate is structural); `gamma` starts at 0 and
moves off it (γ is a true fp32 master parameter, so it is not quantization-locked).

Inference (single-GPU, greedy) from a trained checkpoint:

```bash
python scripts/infer.py --config configs/aeon_350m.yaml --ckpt runs/aeon_350m/ckpt_1000.pt \
    --tokenizer tokenizer/aeon.model --prompt "Aeon" --max-new-tokens 64
```

`infer.py` rebuilds the model from Aeon's own config, applies the training
precision rules (`model.recursion.float()` after the bf16 cast), and — with a
tokenizer — encodes the text prompt and decodes the generation back to text.
Without `--tokenizer` it operates on raw ids (`--prompt-ids "2 …"`, 2 = `<bos>`).

## Precision notes (baked into the code)

- **Recursion stays fp32** — protects the σ-certificate (Cayley solve / SVD).
- **γ is an fp32 master parameter**, re-cast after any global dtype cast — a bf16
  γ has ULP above the optimizer step near 2^-5 and freezes at 1/32.
- **`inject()` adds in fp32** — keeps γ's gradient path fp32 end-to-end.
- **Rotary `inv_freq` is computed fresh in fp32 each forward** — never a buffer a
  cast could degrade.
- **Substrate state follows the parameter dtype** — fp32 state vs bf16 params
  crashes the read matmul.
- **`write_proj` is randomly initialized** — with both γ=0 and `write_proj`=0 the
  write path is gradient-dead; random `write_proj` + γ=0 keeps the start
  contribution zero while letting γ learn.
- **Feedback gate scalars (α, θ) are fp32 master parameters** — a learned bf16
  θ≈0.5 has ULP above the optimizer step and would freeze (the same trap γ hit);
  re-cast to fp32 after the global dtype cast. `W_stressed` trains in bf16.

## Adaptive feedback control

The `matrix_cell` substrate runs a closed loop on its own readout
(`aeon/substrate/feedback.py`): it **senses** load `L(t)` (an EWMA of the
readout's per-step rate of change — cheap, bounded), **gates** on it
(`g(L)=sigmoid(α(L−θ))`, smooth and in [0,1]), and **acts** by blending a
stressed transform into the output:

```
output = (1 − g)·base + g·(output_bound · tanh(W_stressed · base))
```

At low load `g≈0` and the output is exactly the plain readout (the extension
reduces cleanly to prior behaviour). Under stress `g→1` and the output direction
sharpens so Recursion drives a correction back into both streams. The blend is a
convex combination of two `output_bound`-bounded signals, so it is **bounded
elementwise in every mode** — the port's bounded-output contract and Recursion's
σ<margin certificate hold gate-off, gate-on, and mid-transition. Only the
*direction* of the substrate's output changes under stress, never its magnitude.
The gate (`α`, `θ`) is learned. A **minimal auxiliary loss** `L_aux = β·mean g(L)`
(β=0.01, `train.aux_gate_penalty`) penalises gate firing so the gate must *justify*
itself by cutting the primary loss — it prescribes nothing about what firing should
accomplish. `substrate.load()` / `substrate.gate()` expose the live signals.

### Diagnostics (fault isolation)

The extension's correctness is defined by five diagnostics — one per component of
the loop — that isolate *which* part fails rather than reporting one opaque score
(`aeon/diagnostics.py`, run on any checkpoint via `scripts/diagnose_feedback.py`):

| # | component | diagnostic | passes when |
|---|---|---|---|
| 1 | load sensor | `sensor_correlation` | `L(t)` tracks input complexity (and not merely length) |
| 2 | signal gate | `gate_response` | gate is a real threshold, θ inside the observed-load range |
| 3 | actuator | `signal_divergence` | stressed output is directionally distinct from normal (not a scaled copy) |
| 4 | plant | `plant_response` | output distribution shifts under stressed conditioning, beyond matched noise |
| 5 | loop closure | `loop_closure` | load falls after the gate fires, beyond the non-fire baseline |

Each reports `pass` / `fail` / `inconclusive`. On an untrained checkpoint only the
unlearned parts are conclusive (sensor passes; plant/loop are `inconclusive` until
γ lifts and the gate fires) — the diagnostics report that honestly instead of
false-failing. The decision logic of every diagnostic is itself unit-tested against
controlled pass/fail scenarios (`tests/test_feedback_diagnostics.py`).

## Tests

```bash
pip install -e ".[dev]"
# preservation / architectural contract (E0–E1)
python tests/test_substrate_port.py         # substrate port conformance
python tests/test_aeon_sanity.py            # shapes, certificate, gradient flow, determinism,
                                            # γ-updates (bf16-trap regression), no external lib in forward
python tests/test_six_patches.py            # one test per V0.02.02 debug patch
python tests/test_recursion_topology.py     # K=16, fp32 Recursion, single broadcast, certificate
python tests/test_stream_independence.py    # substrate autonomy, no cross-stream imports/reads
python tests/test_config_invariants.py      # no config can silently drift K, dtype, dual-broadcast
python tests/test_tokenizer.py              # tokenizer train + round-trip, special ids, corpus reader
python tests/test_feedback.py               # adaptive feedback: load bound, gate range/grad,
                                            # gate-off reduction, bounded stressed mode, certificate all modes
python tests/test_feedback_diagnostics.py   # the 5 feedback fault-isolation diagnostics
# observability + checkpoint + offline (E2–E4)
python tests/test_observability.py          # equivalence, sampling clock invariance, overhead ceiling
python tests/test_checkpoint.py             # atomic save, resume equivalence, reject-incompatible
python tests/test_diagnose.py               # offline entry point does not mutate checkpoints
```

Full suite: **61 / 61 pass**. Substrate port conformance runs without torch
(contract + AST checks); the model + observability + checkpoint tests require
torch; tokenizer tests require sentencepiece — each skips cleanly when its
backend is absent.

## V0.02.03 architecture-preserving efficiency upgrade (E0–E7)

The repository has completed the eight-phase upgrade specified by the
V0.02.03 execution directive. Every preservation invariant is now backed by
a named test, observability + checkpoint hardening are integrated, and the
primary campaign config is versioned. See `docs/`:

- `E0_REPOSITORY_AUDIT.md`, `TOPOLOGY_MAP.md`, `PRESERVATION_MANIFEST.md` — audit + invariants (E0).
- `E5_CERTIFICATION.md` — bounded runtime certification (E5).
- `OPERATIONS.md` — launch, resume, and recovery for the primary run (E6).
- `SECURITY_MODEL.md` — local security posture (E7).
- `PROXY_CAMPAIGN_PLAN.md` — small-scale proxy comparison plan (E7 / §14).
- `COMMIT_REPORT.md` — the E0–E7 commit ledger (E7).
- `e7_final_evidence.json` — machine-readable evidence bundle.

Efficiency-claim boundaries (§17):
Aeon is architecturally designed for bounded long-range integration through
two parallel streams and a contractive slow-clock Recursion mechanism. Its
efficiency is being measured at the current implementation scale on
laptop-class CPU hardware. Frontier / full-scale / FLOP-based claims are OUT
OF BOUNDS at this stage.

## Status

The integration architecture — substrate port, Recursion joiner with its
certificate, multi-source coupling, and the fp32-γ / fp32-Recursion training
pattern — is exercised end-to-end, and the **from-scratch pipeline is wired and
proven at small scale**: Aeon-trained tokenizer → tokenized-corpus training (loss
decreasing, certificate holding, γ lifting) → checkpoint → text inference. The
350M multilingual prototype config is set (128k vocab, adaptive feedback on with
the β=0.01 gate penalty; 350.28M trainable), the closed-loop control engages under
load in practice (gate ≈0 at low load, ≈1 under stress) with the certificate
holding in every mode, and the **five fault-isolation diagnostics** ship ready to
run on the first checkpoint. The remaining external input is **Dylan's curated
corpus**; when it lands, the small sanity run precedes the full single-epoch run.

`reference/` holds **sealed exploratory background** — not part of Aeon.
