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
pip install -e .            # safetensors, sentencepiece, pyyaml
```

## Run

```bash
python scripts/train.py --config configs/aeon_v1.yaml
```

Everything is random-initialized. The config ships with a **synthetic random-token
data source** so the full pipeline runs end-to-end (forward / loss / backward /
optimizer step / certificate audit / checkpoint, resumable). A real training run
needs a real corpus and an Aeon tokenizer — that is the next step.

Healthy audit lines look like:
`[step N] loss=… sigma_Wh=… sigma_Wc=… holds=True lambda=… gamma=…` — `holds`
stays `True` every step (the certificate is structural); `gamma` starts at 0 and
moves off it (γ is a true fp32 master parameter, so it is not quantization-locked).

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
python tests/test_aeon_sanity.py        # forward shapes, certificate, gradient flow, determinism
```

Substrate port conformance runs without torch (contract + AST checks); the rest
require torch and skip cleanly otherwise.

## Status

The integration architecture — substrate port, Recursion joiner with its
certificate, multi-source coupling, and the fp32-γ / fp32-Recursion training
pattern — has been exercised end-to-end. The transformer side is Aeon-original
and random-initialized. A from-scratch training run on a real corpus (corpus,
tokenizer, scale, compute) is the next step.

`reference/` holds **sealed exploratory background** — not part of Aeon.
