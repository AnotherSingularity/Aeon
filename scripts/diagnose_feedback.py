#!/usr/bin/env python3
"""
scripts/diagnose_feedback.py — run the five feedback-control diagnostics on a
trained Aeon checkpoint.

Fault isolation, not a single score: each component (sensor / gate / actuator /
plant / loop) passes or fails independently, so the pattern tells you exactly
what to fix. See aeon/diagnostics.py for the pass/fail criteria.

    python scripts/diagnose_feedback.py --config configs/aeon_350m.yaml \\
        --ckpt runs/aeon_350m/ckpt_1000.pt --seq-len 128
"""
import argparse

import yaml
import torch

from aeon.diagnostics import run_all
from infer import build_model, _DTYPES   # scripts/ is on sys.path when run as a script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    mcfg = cfg["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _DTYPES[mcfg.get("dtype", "bfloat16")]

    model, tcfg_model = build_model(mcfg, dtype, device)
    blob = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(blob.get("model", blob))
    print(f"[diag] {args.ckpt} (step {blob.get('step', '?')}) "
          f"γ={model.transformer.gamma.item():.4e} device={device}\n")

    results = run_all(model, tcfg_model.vocab_size, seq_len=args.seq_len,
                      device=device, seed=args.seed)
    print("Feedback-control diagnostics (fault isolation):")
    for r in results:
        print(r)

    n_pass = sum(r.status == "pass" for r in results)
    n_fail = sum(r.status == "fail" for r in results)
    n_inc = sum(r.status == "inconclusive" for r in results)
    print(f"\n  {n_pass} pass, {n_fail} fail, {n_inc} inconclusive")
    if n_fail:
        print("  -> failing components isolate what to fix (see aeon/diagnostics.py).")


if __name__ == "__main__":
    main()
