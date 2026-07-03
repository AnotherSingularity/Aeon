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
| **Substrate** | `aeon/substrate/` | the recurrent signal source behind a read/write/cadence **port**. Two cells: `matrix_cell` (matrix state, per-channel decay, outer-product write) and `vector_cell` (single-vector state). Cell choice is deployment-time config via `make_substrate()`. `verify_substrate()` is the conformance gate. |
| **Transformer side** | `aeon/transformer.py` | Aeon's own transformer: RMSNorm, rotary embeddings, grouped-query attention, SwiGLU MLP, pre-norm decoder stack, tied head. Read surface (hidden→manifold) and γ-gated write surface (manifold→hidden). |
| **Coupling** | `aeon/hybrid.py` | slow-clock Recursion (`K`-token windows): running-mean window aggregation, hold-and-broadcast of the previous window's state to the substrate input and the transformer inject, truncated BPTT at window boundaries. |

## Install (CUDA 12.4)

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -e .            # safetensors, pyyaml, numpy
```

## Model scale

`configs/aeon_350m.yaml` is the from-scratch prototype target: **~350.0M
trainable** (hidden 1024, 24 layers, 16 heads × 64, GQA with 4 KV heads,
intermediate 2048, substrate/manifold `h_rec`=512, slow clock K=16, **128k
multilingual vocab**). Only the vocab drives the size vs the earlier 251.7M build:
everything-else is 218.94M unchanged; the 128k×1024 tied embedding is 131.07M.
`configs/aeon_v1.yaml` is a smaller smoke config. Everything is random-initialized.

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

## Tests

```bash
pip install -e ".[dev]"
python tests/test_substrate_port.py     # substrate port conformance
python tests/test_aeon_sanity.py        # shapes, certificate, gradient flow, determinism,
                                        # γ-updates (bf16-trap regression), no external lib in forward
python tests/test_tokenizer.py          # tokenizer train + round-trip, special ids, corpus reader
```

Substrate port conformance runs without torch (contract + AST checks); the model
sanity tests require torch and the tokenizer tests require sentencepiece — each
skips cleanly when its backend is absent.

## Status

The integration architecture — substrate port, Recursion joiner with its
certificate, multi-source coupling, and the fp32-γ / fp32-Recursion training
pattern — is exercised end-to-end, and the **from-scratch pipeline is wired and
proven at small scale**: Aeon-trained tokenizer → tokenized-corpus training (loss
decreasing, certificate holding, γ lifting) → checkpoint → text inference. The
350M multilingual prototype config is set (128k vocab). The remaining external
input is **Dylan's curated corpus**; when it lands, the small sanity run precedes
the full single-epoch run.

`reference/` holds **sealed exploratory background** — not part of Aeon.
