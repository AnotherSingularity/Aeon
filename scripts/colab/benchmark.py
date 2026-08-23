"""scripts/colab/benchmark.py — short fixed-token benchmark of the Aeon
forward+backward path on the current GPU.

Prints:
  * tokens processed
  * wall seconds
  * projected tokens/hour
  * estimated sessions required at 12h/session for a target-tokens goal
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--tokens", type=int, default=50_000,
                    help="approximate number of tokens to run through fwd+bwd")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--target-tokens", type=int, default=100_000_000,
                    help="Stage-1 target for the projection")
    ap.add_argument("--hours-per-session", type=float, default=12.0)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))

    import torch
    if not torch.cuda.is_available():
        print("HALT: CUDA unavailable", file=sys.stderr); return 5
    device = torch.device("cuda")

    from scripts.colab.train_stage import _build_model_and_tokenizer

    model, tok = _build_model_and_tokenizer(root)
    # Fresh untrained-shape params are OK for the benchmark; do NOT load P2
    # here to keep the benchmark independent of checkpoint location.
    model = model.to(device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup 3 steps
    def _rand_batch():
        return torch.randint(low=1, high=tok.vocab_size,
                              size=(args.batch_size, args.seq_len),
                              dtype=torch.long, device=device)
    for _ in range(3):
        ids = _rand_batch()
        optim.zero_grad(set_to_none=True)
        out = model(input_ids=ids)
        loss = out.logits.float().mean()
        loss.backward()
        optim.step()
    torch.cuda.synchronize()

    # Time N steps
    tokens_per_step = args.batch_size * args.seq_len
    n_steps = max(1, args.tokens // tokens_per_step)
    t0 = time.time()
    for _ in range(n_steps):
        ids = _rand_batch()
        optim.zero_grad(set_to_none=True)
        out = model(input_ids=ids)
        loss = out.logits.float().mean()
        loss.backward()
        optim.step()
    torch.cuda.synchronize()
    wall = time.time() - t0
    total_tokens = n_steps * tokens_per_step
    tps = total_tokens / wall
    tokens_per_hour = tps * 3600
    tokens_per_session = tokens_per_hour * args.hours_per_session
    sessions_needed = args.target_tokens / max(1.0, tokens_per_session)

    report = {
        "device": torch.cuda.get_device_name(0),
        "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / (1 << 30), 2),
        "torch_version": torch.__version__,
        "cuda_version_torch_built_with": torch.version.cuda,
        "steps": n_steps,
        "tokens_processed": total_tokens,
        "wall_seconds": round(wall, 2),
        "tokens_per_second": round(tps, 1),
        "tokens_per_hour": int(tokens_per_hour),
        "hours_per_session": args.hours_per_session,
        "tokens_per_session": int(tokens_per_session),
        "target_tokens": args.target_tokens,
        "estimated_sessions": round(sessions_needed, 2),
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
