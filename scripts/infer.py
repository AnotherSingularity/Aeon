#!/usr/bin/env python3
"""
scripts/infer.py — Aeon single-GPU inference (greedy generation).

Loads an Aeon checkpoint and its YAML config, rebuilds the model from Aeon's own
config (NOTHING external — no external config.json, no external checkpoint, no
external tokenizer), and generates greedily.

PRECISION (must match training): compute dtype is bf16, EXCEPT Recursion, which
is kept fp32 — `model.recursion.float()` after the dtype cast. Recursion carries
the σ<margin certificate and its Cayley solve / SVD have no bf16 path; a bf16
Recursion would break the certificate. γ is likewise re-cast to fp32 for parity
with the trained master parameter.

TOKENIZER (decision pending — see README, REBUILD §2):
  * Option A — an Aeon tokenizer trained from scratch on the corpus.
  * Option B — a public-domain tokenizer with no model affiliation.
Neither is wired yet (out of scope until the corpus is ready). Until one lands,
this script operates directly on integer token ids: pass `--prompt-ids "1 2 3"`,
or omit it to seed from a single BOS-like id 1. A UTF-8 *byte* fallback is offered
only as a pipeline smoke (`--bytes`), clamped into the vocab — it is NOT Aeon's
tokenizer and produces no meaningful text; it only exercises the generate loop.
"""
import argparse

import yaml
import torch

from aeon.hybrid import HybridModel
from aeon.transformer import AeonTransformerConfig

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def build_model(mcfg, dtype, device):
    """Rebuild the model exactly as scripts/train.py does (same construction, so a
    training checkpoint loads cleanly), then apply the inference precision rules."""
    tcfg_model = AeonTransformerConfig(**mcfg.get("transformer", {}))
    model = HybridModel(
        h_rec=mcfg["h_rec"], K=mcfg["K"], transformer_config=tcfg_model,
        substrate=mcfg.get("substrate"), margin_h=mcfg["margin_h"],
        margin_c=mcfg["margin_c"], freeze_backbone=mcfg.get("freeze_backbone", False),
        use_embedding_input=mcfg.get("use_embedding_input", True), dtype=dtype,
    ).to(device)
    model.to(dtype=dtype)        # cast compute path to the run dtype...
    model.recursion.float()      # ...except Recursion (fp32 σ-certificate)
    model.transformer.gamma.data = model.transformer.gamma.data.float()  # fp32 master γ
    model.eval()
    return model, tcfg_model


@torch.no_grad()
def generate(model, input_ids, max_new_tokens, vocab_size, eos_id=None):
    """Greedy autoregressive generation. No KV cache: each step re-runs the full
    forward (fine for single-GPU smoke-scale generation); Recursion state is
    rebuilt inside each forward, so generation is stateless across steps."""
    ids = input_ids
    for _ in range(max_new_tokens):
        logits = model(input_ids=ids).logits            # (B, T, V)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1) greedy
        ids = torch.cat([ids, next_id], dim=1)
        if eos_id is not None and (next_id == eos_id).all():
            break
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Aeon YAML config (same as training)")
    ap.add_argument("--ckpt", required=True, help="Aeon checkpoint (.pt from scripts/train.py)")
    ap.add_argument("--prompt-ids", default="1",
                    help="space-separated integer token ids to seed with (tokenizer pending)")
    ap.add_argument("--bytes", dest="byte_prompt", default=None,
                    help="SMOKE ONLY: UTF-8 bytes of this string, clamped into vocab "
                         "(not Aeon's tokenizer; produces no meaningful text)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--eos-id", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mcfg = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _DTYPES[mcfg.get("dtype", "bfloat16")]

    model, tcfg_model = build_model(mcfg, dtype, device)
    vocab = tcfg_model.vocab_size

    blob = torch.load(args.ckpt, map_location=device)
    state = blob.get("model", blob)
    model.load_state_dict(state)
    print(f"[infer] loaded {args.ckpt} (step {blob.get('step', '?')}) | "
          f"device={device} dtype={dtype} gamma={model.transformer.gamma.item():.4e}")

    if args.byte_prompt is not None:
        print("[infer] WARNING: --bytes is a pipeline smoke, NOT Aeon's tokenizer; "
              "output is not meaningful text (tokenizer decision pending).")
        seed = [b % vocab for b in args.byte_prompt.encode("utf-8")] or [1]
    else:
        seed = [int(x) for x in args.prompt_ids.split()]
    if any(not (0 <= i < vocab) for i in seed):
        raise ValueError(f"seed ids must be in [0, {vocab}); got {seed}")

    input_ids = torch.tensor([seed], dtype=torch.long, device=device)
    out = generate(model, input_ids, args.max_new_tokens, vocab, eos_id=args.eos_id)
    print(f"[infer] generated ids: {out[0].tolist()}")


if __name__ == "__main__":
    main()
