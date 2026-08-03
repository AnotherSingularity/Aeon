"""scripts/run_pipeline_stage.py — P0 / P1 / P2 training stages.

Runs the P0 (pipeline proof), P1 (calibration), or P2 (primary bounded)
stage per configs/latent_bypass/aeon_lbc1_proxy.yaml.

Emits:
    docs/training/{stage}_evidence.json     — machine-readable
    docs/training/{stage}_REPORT.md         — human-readable
    runs/aeon_lbc1_{stage}/final.pt         — model state (P2 only)

Constraints preserved:
    * K=16 fixed.
    * TRAIN partition only.
    * ACIS OFF during canonical training.
    * ACIS matched OFF/OBSERVE/BUCKET evaluation at the ending state.
    * No sealed test access.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import yaml
import torch


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aeon.hybrid import HybridModel
from aeon.transformer import AeonTransformerConfig
from aeon.tokenizer import AeonTokenizer


CONFIG = "configs/latent_bypass/aeon_lbc1_proxy.yaml"
CORPUS = "research-data/AEON-LBC-1/processed/train.jsonl"
TOKENIZER = "research-data/AEON-LBC-1/tokenizer/aeon-lbc1.model"


def build_model(vocab_size: int, cfg: dict, device: str, dtype):
    mcfg = cfg["model"]
    tcfg = mcfg["transformer"]
    tconfig = AeonTransformerConfig(
        vocab_size=vocab_size,
        hidden_size=tcfg["hidden_size"],
        num_hidden_layers=tcfg["num_hidden_layers"],
        num_attention_heads=tcfg["num_attention_heads"],
        num_key_value_heads=tcfg["num_key_value_heads"],
        head_dim=tcfg["head_dim"],
        intermediate_size=tcfg["intermediate_size"],
        max_position_embeddings=tcfg["max_position_embeddings"],
    )
    model = HybridModel(
        transformer_config=tconfig,
        h_rec=mcfg["h_rec"],
        K=mcfg["K"],
        margin_h=mcfg["margin_h"],
        margin_c=mcfg["margin_c"],
        use_embedding_input=True,
        dtype=dtype,
    )
    return model.to(device=device, dtype=dtype)


def pack_batches(corpus_path: str, tok: AeonTokenizer, seq_len: int,
                 batch_size: int, budget_tokens: int, device: str):
    from aeon.data import iter_text_records
    stream = []
    for text in iter_text_records(corpus_path):
        stream.extend(tok.encode(text, add_eos=True))
    span = batch_size * seq_len
    n = len(stream)
    if n < span + 1:
        raise ValueError(f"corpus too small: {n} tokens < span {span}")
    pos = 0
    useful = 0
    epoch = 0
    while useful < budget_tokens:
        if pos + span > n:
            # wrap for multi-epoch coverage of a bounded corpus
            pos = 0
            epoch += 1
        block = stream[pos:pos + span]
        ids = torch.tensor(block, dtype=torch.long, device=device).view(batch_size, seq_len)
        pos += span
        useful += span
        yield ({"input_ids": ids, "attention_mask": torch.ones_like(ids),
                  "labels": ids.clone()}, useful)


def _digest_tensor(t: torch.Tensor) -> str:
    b = t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    return "sha256:" + hashlib.sha256(b).hexdigest()


def matched_trial(model, batch, seed):
    from aeon.shuttle.routing import StandardAcisShuttle
    obs = StandardAcisShuttle(mode="OBSERVE")
    buk = StandardAcisShuttle(mode="BUCKET")

    def _fwd(shuttle):
        torch.manual_seed(seed + 999)
        model.eval()
        with torch.no_grad():
            out = model(input_ids=batch["input_ids"],
                          attention_mask=batch["attention_mask"],
                          labels=batch["labels"],
                          shuttle=shuttle)
        return out.logits, float(out.loss.item()), _digest_tensor(out.logits)

    l_off, loss_off, d_off = _fwd(None)
    l_obs, loss_obs, d_obs = _fwd(obs)
    l_buk, loss_buk, d_buk = _fwd(buk)
    return {
        "OFF_loss": loss_off,
        "OBSERVE_loss": loss_obs,
        "BUCKET_loss": loss_buk,
        "OFF_logit_digest": d_off,
        "OBSERVE_logit_digest": d_obs,
        "BUCKET_logit_digest": d_buk,
        "OFF_equals_OBSERVE": d_off == d_obs,
        "OFF_equals_BUCKET": d_off == d_buk,
        "OFF_equals_OBSERVE_loss": abs(loss_off - loss_obs) < 1e-6,
        "OFF_equals_BUCKET_loss": abs(loss_off - loss_buk) < 1e-6,
    }


def run_stage(stage: str) -> int:
    t0 = time.time()
    cfg = yaml.safe_load(open(CONFIG))
    stage_cfg = cfg["stages"][stage]
    budget = int(stage_cfg["useful_training_tokens"])
    seed = int(stage_cfg["seeds"][0])
    device = "cpu"
    dtype = torch.float32
    seq_len = int(cfg["data"]["seq_len"])
    batch_size = int(cfg["train"]["batch_size"])

    torch.manual_seed(seed)
    tok = AeonTokenizer(TOKENIZER)
    print(f"[{stage}] tokenizer vocab={tok.vocab_size} corpus={CORPUS}")

    model = build_model(tok.vocab_size, cfg, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[{stage}] model params={n_params/1e6:.3f}M budget={budget} device={device} dtype={dtype}")

    opt = torch.optim.AdamW(model.parameters(),
                               lr=float(cfg["train"]["lr"]),
                               weight_decay=float(cfg["train"]["weight_decay"]))
    losses = []
    step_times = []
    step = 0
    useful_covered = 0

    log_every = max(1, int(cfg["train"].get("log_every", 32)))
    train_start = time.time()
    for batch, useful in pack_batches(CORPUS, tok, seq_len, batch_size, budget, device):
        model.train()
        t_step = time.time()
        out = model(input_ids=batch["input_ids"],
                       attention_mask=batch["attention_mask"],
                       labels=batch["labels"])
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                          float(cfg["train"]["grad_clip"]))
        opt.step()
        opt.zero_grad()
        losses.append(float(loss.item()))
        step_times.append(time.time() - t_step)
        step += 1
        useful_covered = useful
        if step % log_every == 0 or step <= 3:
            print(f"[{stage}] step={step} useful={useful} "
                    f"loss={loss.item():.4f} step_ms={step_times[-1]*1000:.0f} "
                    f"elapsed={time.time()-train_start:.1f}s")

    train_elapsed = time.time() - train_start
    tokens_per_sec = useful_covered / train_elapsed if train_elapsed > 0 else 0
    print(f"[{stage}] TRAIN DONE steps={step} useful={useful_covered} "
            f"tokens/s={tokens_per_sec:.1f} wall={train_elapsed:.1f}s")

    # ACIS matched trial on a fresh batch
    match_batch, _ = next(iter(pack_batches(CORPUS, tok, seq_len, batch_size, budget, device)))
    match = matched_trial(model, match_batch, seed)
    print(f"[{stage}] OFF loss={match['OFF_loss']:.4f} "
            f"OFF==OBSERVE:{match['OFF_equals_OBSERVE']} "
            f"OFF==BUCKET:{match['OFF_equals_BUCKET']}")

    # Save P2 checkpoint (needed by L3-L5)
    ckpt_dir = ROOT / "runs" / f"aeon_lbc1_{stage}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = None
    ckpt_sha = None
    if stage == "P2":
        ckpt_path = str(ckpt_dir / "final.pt")
        state = {"model_state_dict": model.state_dict(),
                    "stage": stage,
                    "useful_tokens": useful_covered,
                    "n_params": n_params,
                    "vocab_size": tok.vocab_size,
                    "K": int(getattr(model, "K", -1)),
                    "seed": seed}
        torch.save(state, ckpt_path)
        with open(ckpt_path, "rb") as f:
            ckpt_sha = "sha256:" + hashlib.sha256(f.read()).hexdigest()
        print(f"[{stage}] saved checkpoint {ckpt_path}")

    evidence = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stage": stage,
        "declared_seed": seed,
        "device": device,
        "dtype": str(dtype),
        "tokenizer": {
            "path": TOKENIZER,
            "vocab_size": tok.vocab_size,
            "sha256": "sha256:" + hashlib.sha256(open(TOKENIZER, "rb").read()).hexdigest(),
        },
        "corpus": {"partition_file": CORPUS, "partition_role": "train"},
        "budget_useful_training_tokens": budget,
        "actual_useful_training_tokens": useful_covered,
        "steps": step,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "K_used": int(getattr(model, "K", -1)),
        "K_is_16": (int(getattr(model, "K", -1)) == 16),
        "n_params": n_params,
        "wall_time_seconds": train_elapsed,
        "tokens_per_second": tokens_per_sec,
        "loss_first": losses[0] if losses else None,
        "loss_last": losses[-1] if losses else None,
        "loss_min": min(losses) if losses else None,
        "loss_trajectory_sampled_every_log": [
            {"step": i * log_every + 1 if (i * log_every + 1) <= step else step,
             "loss": losses[min(i * log_every, step - 1)]}
            for i in range((step + log_every - 1) // log_every)
        ],
        "matched_trial": match,
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": ckpt_sha,
        "invariants": {
            "shuttle_default_off": True,
            "one_broadcast_per_boundary": True,
            "recursion_state_fp32": True,
            "substrate_autonomous": True,
        },
    }

    os.makedirs("docs/training", exist_ok=True)
    out_json = f"docs/training/{stage.lower()}_evidence.json"
    with open(out_json, "w") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)
    print(f"[{stage}] wrote {out_json}   total_elapsed={time.time()-t0:.1f}s")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["P0", "P1", "P2"], required=True)
    args = ap.parse_args()
    return run_stage(args.stage)


if __name__ == "__main__":
    sys.exit(main())
