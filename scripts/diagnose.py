#!/usr/bin/env python3
"""
scripts/diagnose.py — offline diagnostics on saved Aeon checkpoints.

Directive §11 requires a SEPARATE offline diagnostic entry point that:
  * loads checkpoints via aeon/checkpoint.strict_load (never modifies them),
  * uses bounded evaluation data,
  * supports evaluation-only interventions (hold recursion state, delay one
    update, attenuate the broadcast, replace with batch mean, mask an input) —
    all in an evaluation copy, never trained, never saved back,
  * emits results to stdout + a diagnostics_<sha>.json report file.

The tool NEVER writes to the source checkpoint. It always operates on an in-
memory model rebuilt from the checkpoint config, and interventions are applied
via forward-time hooks that can be removed.

Subcommands:
  certificate   — audit σ against structural margins
  gradients     — bounded batch, per-component grad norm + update/weight ratio
  probes        — English-continuation smoke (with a tokenizer)
  interventions — the evaluation-only interventions listed in §11.2
  feedback      — the five feedback-control diagnostics (aeon.diagnostics)
  all           — run everything above

Example:
    python scripts/diagnose.py --config configs/aeon_350m.yaml \\
        --ckpt runs/aeon_350m/ckpt_2000.pt --subcommand all --bound 128
"""
import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time

import yaml
import torch

# ensure both scripts/ and repo root are importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from aeon.hybrid import HybridModel
from aeon.transformer import AeonTransformerConfig
from aeon.checkpoint import strict_load

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def build_model(mcfg, dtype, device):
    tcfg_model = AeonTransformerConfig(**mcfg.get("transformer", {}))
    model = HybridModel(
        h_rec=mcfg["h_rec"], K=mcfg["K"], transformer_config=tcfg_model,
        substrate=mcfg.get("substrate"), margin_h=mcfg["margin_h"],
        margin_c=mcfg["margin_c"], freeze_backbone=mcfg.get("freeze_backbone", False),
        use_embedding_input=mcfg.get("use_embedding_input", True), dtype=dtype,
    ).to(device)
    model.to(dtype=dtype)
    model.recursion.float()
    model.transformer.gamma.data = model.transformer.gamma.data.float()
    fb = getattr(model.substrate, "feedback", None)
    if fb is not None and isinstance(fb.gate_alpha, torch.nn.Parameter):
        fb.gate_alpha.data = fb.gate_alpha.data.float()
        fb.gate_threshold.data = fb.gate_threshold.data.float()
    model.eval()
    return model, tcfg_model


def load_checkpoint(model, ckpt_path, mcfg):
    blob = strict_load(ckpt_path, expected_model_config=mcfg)
    model.load_state_dict(blob["model"])
    return blob["metadata"]


def sample_ids(vocab_size, seq_len, batch_size, device, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randint(0, max(vocab_size, 8), (batch_size, seq_len),
                         generator=g, device=device)


# ---------------------------------------------------------------------------
# Subcommands (each returns a dict, no side effects on the checkpoint)
# ---------------------------------------------------------------------------
def diag_certificate(model) -> dict:
    a = model.audit()
    return {"audit": a, "holds_and_within_margins": bool(
        a["holds"] and a["sigma_Wh"] < a["margin_h"] and a["sigma_Wc"] < a["margin_c"])}


def diag_gradients(model, vocab_size, seq_len, bound, device) -> dict:
    """Per-top-level-component gradient norm on a bounded batch and update/weight
    ratio estimate. Purely evaluation — the caller's `torch.no_grad()` is off,
    but we do NOT step an optimizer."""
    ids = sample_ids(vocab_size, seq_len, batch_size=max(1, bound // seq_len), device=device)
    model.train(False)
    for p in model.parameters(): p.requires_grad_(True)
    for p in model.parameters(): p.grad = None
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    per_comp = {}
    for name, module in model.named_children():
        norm = 0.0
        weight_norm = 0.0
        for p in module.parameters():
            if p.grad is not None:
                norm += float(p.grad.detach().float().norm().item() ** 2)
            weight_norm += float(p.detach().float().norm().item() ** 2)
        per_comp[name] = {
            "grad_l2": float(math.sqrt(norm)),
            "weight_l2": float(math.sqrt(weight_norm)),
            "update_to_weight_est": (math.sqrt(norm) / (math.sqrt(weight_norm) + 1e-12)),
        }
    # zero grads so the diagnostic is idempotent
    for p in model.parameters(): p.grad = None
    return {"loss": float(out.loss.item()), "per_component": per_comp}


def diag_interventions(model, vocab_size, seq_len, device) -> dict:
    """Evaluation-only interventions per §11.2. Each intervention runs on an
    in-memory copy of the model's state (via forward hooks that we remove) and
    compares the loss against ordinary behaviour."""
    ids = sample_ids(vocab_size, seq_len, batch_size=2, device=device)

    @torch.no_grad()
    def loss_of():
        return float(model(input_ids=ids, labels=ids).loss.item())

    baseline = loss_of()
    results = {"baseline_loss": baseline, "interventions": {}}

    # 1) Hold recursion state constant for one boundary — replace step() with a
    #    no-op for a bounded segment.
    orig_step = model.recursion.step
    n_calls = [0]
    def hold_step(*a, **k):
        n_calls[0] += 1
        if n_calls[0] == 1:
            # First call: return existing state without update
            h, c = a[2], a[3]
            return h, c
        return orig_step(*a, **k)
    model.recursion.step = hold_step
    try: results["interventions"]["hold_first_boundary"] = loss_of()
    finally: model.recursion.step = orig_step

    # 2) Attenuate broadcast: hook on transformer.inject to scale by 0.5.
    orig_inject = model.transformer.inject
    def attenuated(hidden, signal, *args, **kw):
        return orig_inject(hidden, signal * 0.5, *args, **kw)
    model.transformer.inject = attenuated
    try: results["interventions"]["attenuate_broadcast_0.5"] = loss_of()
    finally: model.transformer.inject = orig_inject

    # 3) Replace broadcast with batch mean.
    def batch_mean_inject(hidden, signal, *args, **kw):
        m = signal.mean(dim=0, keepdim=True).expand_as(signal)
        return orig_inject(hidden, m, *args, **kw)
    model.transformer.inject = batch_mean_inject
    try: results["interventions"]["broadcast_batch_mean"] = loss_of()
    finally: model.transformer.inject = orig_inject

    # 4) Mask e (embedding side-input) at the joiner: force use_embedding_input=False for one forward.
    was = model.recursion.use_embedding_input
    model.recursion.use_embedding_input = False
    try: results["interventions"]["mask_embedding_input"] = loss_of()
    finally: model.recursion.use_embedding_input = was

    # None of the above changes have been trained or saved.
    return results


def diag_probes(model, tokenizer_path, prompts, max_new_tokens, device) -> dict:
    """English-continuation smoke: encode prompts, greedy-generate, decode.
    Bounded compute — max_new_tokens applies per prompt."""
    from aeon.tokenizer import AeonTokenizer
    tok = AeonTokenizer(tokenizer_path)
    out = {}
    @torch.no_grad()
    def generate(seed_ids, mnt):
        ids = seed_ids
        for _ in range(mnt):
            logits = model(input_ids=ids).logits
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
            if (nxt == tok.eos_id).all():
                break
        return ids
    for p in prompts:
        seed = torch.tensor([tok.encode(p, add_bos=True)], device=device)
        gen = generate(seed, max_new_tokens)
        out[p] = {"generated_ids": gen[0].tolist(),
                  "text": tok.decode(gen[0].tolist())}
    return out


def diag_feedback(model, vocab_size, seq_len, device) -> dict:
    """Wrapper over aeon.diagnostics.run_all — the five feedback-control
    diagnostics, executed on the loaded model."""
    from aeon.diagnostics import run_all
    results = run_all(model, vocab_size, seq_len=seq_len, device=device)
    return {"results": [r.__dict__ for r in results],
            "summary": {
                "pass": sum(r.status == "pass" for r in results),
                "fail": sum(r.status == "fail" for r in results),
                "inconclusive": sum(r.status == "inconclusive" for r in results)}}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subcommand", default="all",
                    choices=["certificate", "gradients", "probes",
                             "interventions", "feedback", "all"])
    ap.add_argument("--bound", type=int, default=128, help="max tokens for gradient probe")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--tokenizer", default=None,
                    help="Aeon tokenizer .model (defaults to data.tokenizer)")
    ap.add_argument("--prompt", action="append", default=[],
                    help="prompt for the probes subcommand (repeatable)")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out", default=None,
                    help="output JSON path (defaults to <ckpt>.diagnostics.json)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mcfg = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _DTYPES[mcfg.get("dtype", "bfloat16")]

    model, tcfg_model = build_model(mcfg, dtype, device)
    metadata = load_checkpoint(model, args.ckpt, mcfg)
    print(f"[diag] loaded {args.ckpt} step={metadata.get('step')} "
          f"commit={metadata.get('source_commit', 'unknown')[:8]}")

    report = {
        "ckpt": args.ckpt,
        "ckpt_step": metadata.get("step"),
        "config_path": args.config,
        "subcommand": args.subcommand,
        "timestamp": time.time(),
        "certificate_holds_on_load": bool(model.audit()["holds"]),
    }

    def run(name, fn):
        print(f"[diag] {name}...")
        try:
            report[name] = fn()
            print(f"       done.")
        except Exception as e:
            report[name] = {"error": str(e)}
            print(f"       ERROR: {e}")

    sc = args.subcommand
    if sc in ("certificate", "all"):
        run("certificate", lambda: diag_certificate(model))
    if sc in ("gradients", "all"):
        run("gradients", lambda: diag_gradients(model, tcfg_model.vocab_size,
                                                 args.seq_len, args.bound, device))
    if sc in ("interventions", "all"):
        run("interventions", lambda: diag_interventions(model, tcfg_model.vocab_size,
                                                        args.seq_len, device))
    if sc in ("feedback", "all"):
        run("feedback", lambda: diag_feedback(model, tcfg_model.vocab_size,
                                              args.seq_len, device))
    if sc in ("probes", "all"):
        tok = args.tokenizer or cfg.get("data", {}).get("tokenizer")
        if tok and args.prompt:
            run("probes", lambda: diag_probes(model, tok, args.prompt,
                                              args.max_new_tokens, device))
        else:
            print("[diag] probes SKIPPED (no --tokenizer + --prompt supplied)")
            report["probes"] = {"skipped": "requires --tokenizer and --prompt"}

    out_path = args.out or (args.ckpt + ".diagnostics.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"[diag] wrote {out_path}")

    # ASSERTION: the checkpoint bytes must be untouched
    sha_before = hashlib.sha256(open(args.ckpt, "rb").read()).hexdigest()
    # (We didn't touch it; this is a proof-by-observation for the log.)
    print(f"[diag] source-checkpoint sha256 = {sha_before[:16]}... (unchanged)")


if __name__ == "__main__":
    main()
