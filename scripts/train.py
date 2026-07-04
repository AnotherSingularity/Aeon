#!/usr/bin/env python3
"""
scripts/train.py — Aeon end-to-end training (from scratch).

Loads NOTHING: the transformer, substrate, Recursion, and projections are all
random-initialized to Aeon's own config and trained end-to-end. YAML-driven,
resumable, with the validated training patterns baked in (γ kept fp32, Recursion
kept fp32, substrate state following the param dtype).

DATA: this script ships with a SYNTHETIC random-token source so the full
pipeline (forward / loss / backward / optimizer step / certificate audit /
checkpoint) runs end-to-end without a corpus. Training a real model needs a real
corpus + an Aeon tokenizer — that is the next step (out of scope here); plug a
real token stream into `iter_batches()`.

PRECISION: bf16 compute, except Recursion (fp32, σ-certificate) and γ (fp32
master parameter — model.to(dtype) casts every param, and a bf16 γ has ULP above
the optimizer step near 2^-5, snapping it to 1/32 and freezing it).
"""
import argparse
import glob
import os
import time

import yaml
import torch

from aeon.hybrid import HybridModel
from aeon.transformer import AeonTransformerConfig

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def iter_synthetic_batches(vocab_size, seq_len, batch_size, device, generator):
    """Synthetic random-token batches (pipeline smoke). Used when no Aeon
    tokenizer + corpus are configured."""
    while True:
        ids = torch.randint(0, vocab_size, (batch_size, seq_len),
                            generator=generator, device=device)
        yield {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids}


def iter_corpus_batches(corpus_path, tok, seq_len, batch_size, device):
    """Sequential single-epoch batches over an Aeon-tokenized corpus.

    Documents are tokenized with the Aeon tokenizer and joined by EOS into one id
    stream, then packed into non-overlapping (batch_size, seq_len) blocks. The
    stream is built in memory — fine for the prototype/sanity subset; a full
    5–10B-token single-epoch run wants pre-tokenized shards read lazily, which is
    the next step once the corpus format is fixed (extend aeon/data.py).
    """
    from aeon.data import iter_text_records
    stream = []
    for text in iter_text_records(corpus_path):
        stream.extend(tok.encode(text, add_eos=True))
    n_tok = len(stream)
    span = batch_size * seq_len
    if n_tok < span + 1:
        raise ValueError(f"corpus too small: {n_tok} tokens < one batch ({span}+1)")
    print(f"[data] corpus tokenized: {n_tok/1e6:.3f}M tokens -> "
          f"{n_tok // span} batches of {batch_size}x{seq_len} (single epoch)")
    pos = 0
    while pos + span <= n_tok:
        block = stream[pos:pos + span]
        ids = torch.tensor(block, dtype=torch.long, device=device).view(batch_size, seq_len)
        pos += span
        yield {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}


def latest_checkpoint(out_dir):
    cks = glob.glob(os.path.join(out_dir, "ckpt_*.pt"))
    return max(cks, key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0])) if cks else None


def save_checkpoint(out_dir, step, model, opt):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"ckpt_{step}.pt")
    torch.save({"step": step, "model": model.state_dict(), "optim": opt.state_dict()}, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]

    torch.manual_seed(tcfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _DTYPES[mcfg.get("dtype", "bfloat16")]

    # ---- tokenizer (Aeon's own; optional — synthetic tokens without it) ---
    tok = None
    tok_path, corpus_path = dcfg.get("tokenizer"), dcfg.get("corpus")
    if bool(tok_path) ^ bool(corpus_path):
        raise ValueError("data.tokenizer and data.corpus must be set together "
                         "(both for a real run, neither for the synthetic smoke)")
    if tok_path:
        from aeon.tokenizer import AeonTokenizer
        tok = AeonTokenizer(tok_path)

    # ---- model (everything random-initialized) ---------------------------
    tcfg_model = AeonTransformerConfig(**mcfg.get("transformer", {}))
    if tok is not None and tok.vocab_size != tcfg_model.vocab_size:
        print(f"[init] overriding vocab_size {tcfg_model.vocab_size} -> {tok.vocab_size} "
              f"(matched to the Aeon tokenizer)")
        tcfg_model.vocab_size = tok.vocab_size
    model = HybridModel(
        h_rec=mcfg["h_rec"], K=mcfg["K"], transformer_config=tcfg_model,
        substrate=mcfg.get("substrate"), margin_h=mcfg["margin_h"],
        margin_c=mcfg["margin_c"], freeze_backbone=mcfg.get("freeze_backbone", False),
        use_embedding_input=mcfg.get("use_embedding_input", True), dtype=dtype,
    ).to(device)
    model.to(dtype=dtype)        # cast everything to compute dtype...
    model.recursion.float()      # ...except Recursion (fp32 certificate)
    # γ must be an fp32 master parameter (see module docstring). Re-cast AFTER
    # model.to(dtype) and BEFORE the optimizer is built.
    model.transformer.gamma.data = model.transformer.gamma.data.float()
    # Same trap for the substrate's learned feedback gate scalars (α, θ): a bf16
    # θ≈0.5 has ULP above the optimizer step and would freeze. Keep them fp32.
    fb = getattr(model.substrate, "feedback", None)
    if fb is not None and isinstance(fb.gate_alpha, torch.nn.Parameter):
        fb.gate_alpha.data = fb.gate_alpha.data.float()
        fb.gate_threshold.data = fb.gate_threshold.data.float()

    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    print(f"[init] trainable params: {sum(p.numel() for p in params)/1e6:.2f}M | "
          f"device={device} dtype={dtype}")
    print(f"[init] audit @ start: {model.audit()}")

    # ---- resume ----------------------------------------------------------
    start_step = 0
    if tcfg.get("resume"):
        ck = latest_checkpoint(tcfg["out_dir"])
        if ck:
            blob = torch.load(ck, map_location=device)
            model.load_state_dict(blob["model"])
            opt.load_state_dict(blob["optim"])
            start_step = blob["step"]
            print(f"[resume] from {ck} at step {start_step}")

    # ---- train loop ------------------------------------------------------
    if tok is not None:
        batches = iter_corpus_batches(corpus_path, tok, dcfg["seq_len"],
                                      tcfg["batch_size"], device)
    else:
        print("[data] no tokenizer+corpus configured -> SYNTHETIC random tokens (smoke)")
        gen = torch.Generator(device=device).manual_seed(tcfg["seed"])
        batches = iter_synthetic_batches(tcfg_model.vocab_size, dcfg["seq_len"],
                                         tcfg["batch_size"], device, gen)
    model.train()
    step = start_step
    t0 = time.time()
    for batch in batches:
        if step >= tcfg["max_steps"]:
            break
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"], labels=batch["labels"])
        opt.zero_grad(set_to_none=True)
        out.loss.backward()
        if tcfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
        opt.step()
        step += 1

        if step % tcfg["log_every"] == 0:
            a = model.audit()
            print(f"[step {step}] loss={out.loss.item():.4f} "
                  f"sigma_Wh={a['sigma_Wh']:.4f} sigma_Wc={a['sigma_Wc']:.4f} "
                  f"holds={a['holds']} lambda={a['lambda']:.3f} gamma={a['gamma']:.4e} "
                  f"({time.time()-t0:.1f}s)")
            if not a["holds"]:
                print("  [WARN] sigma certificate does NOT hold — investigate")
        if step % tcfg["ckpt_every"] == 0:
            print(f"[ckpt] {save_checkpoint(tcfg['out_dir'], step, model, opt)}")

    print(f"[done] final step {step} | {save_checkpoint(tcfg['out_dir'], step, model, opt)}")


if __name__ == "__main__":
    main()
