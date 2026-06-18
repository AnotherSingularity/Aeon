#!/usr/bin/env python3
"""
scripts/train.py — Stage-1 hybrid training.

⚠️ UNRUN locally (no torch/transformers/HF access in the authoring env). This is
the script Dylan runs on Vast. Single, resumable, YAML-driven loop:

  - loads the R1 backbone + builds substrate + recursion + hybrid (HybridModel)
  - freezes the backbone; trains substrate / recursion / projections / read+write
  - alpaca instruction tuning (Stage-1 parity), bf16, batch=1, seq=512
  - audit logging every `log_every` steps (sigma_Wh, sigma_Wc, holds, gamma, loss)
  - checkpoints every `ckpt_every`; auto-resumes from the latest checkpoint

PRECISION NOTE (Vast-tuning item): the model is bf16 except Recursion, which is
kept fp32 to protect the σ<margin certificate (Cayley solve / svd lack bf16). If
bf16 AdamW on the trainable params proves unstable, switch to fp32 master params
+ autocast — flagged, not pre-optimized.
"""
import argparse
import glob
import os
import random
import time

import yaml
import torch

from aeon.hybrid import HybridModel
from aeon.transformer import config_from_pretrained

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

PROMPT_INPUT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n"
)
PROMPT_NO_INPUT = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Response:\n"
)


class AlpacaDataset(torch.utils.data.Dataset):
    """Alpaca instruction tuning: prompt tokens masked (-100) in labels; the
    sequence is padded/truncated to a fixed seq_len so the K-window count is
    exact (T / K windows)."""

    def __init__(self, split, tokenizer, seq_len):
        self.data = split
        self.tok = tokenizer
        self.seq_len = seq_len
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        tmpl = PROMPT_INPUT if ex.get("input") else PROMPT_NO_INPUT
        prompt = tmpl.format(instruction=ex["instruction"], input=ex.get("input", ""))
        response = ex["output"]

        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        r_ids = self.tok(response, add_special_tokens=False)["input_ids"] + [self.eos_id]

        ids = (p_ids + r_ids)[: self.seq_len]
        labels = ([-100] * len(p_ids) + r_ids)[: self.seq_len]

        pad = self.seq_len - len(ids)
        attn = [1] * len(ids) + [0] * pad
        ids = ids + [self.pad_id] * pad
        labels = labels + [-100] * pad

        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def latest_checkpoint(out_dir):
    cks = glob.glob(os.path.join(out_dir, "ckpt_*.pt"))
    if not cks:
        return None
    return max(cks, key=lambda p: int(os.path.basename(p).split("_")[1].split(".")[0]))


def save_checkpoint(out_dir, step, model, opt):
    os.makedirs(out_dir, exist_ok=True)
    # backbone is frozen + reloaded from R1 on resume; don't checkpoint it.
    trainable = {k: v for k, v in model.state_dict().items()
                 if not k.startswith("transformer.backbone")}
    path = os.path.join(out_dir, f"ckpt_{step}.pt")
    torch.save({"step": step, "model": trainable, "optim": opt.state_dict()}, path)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]

    random.seed(tcfg["seed"]); torch.manual_seed(tcfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = _DTYPES[mcfg.get("dtype", "bfloat16")]

    # ---- checkpoint dir (local path, or download via huggingface_hub) ----
    ckpt_dir = mcfg.get("checkpoint_dir")
    if not ckpt_dir:
        from huggingface_hub import snapshot_download   # I/O only, not architecture
        ckpt_dir = snapshot_download(mcfg["model_name"])
    qcfg = config_from_pretrained(ckpt_dir)              # Aeon config from config.json
    print(f"[init] Aeon Qwen2 config: {qcfg}")

    # ---- model (Aeon-original transformer; R1 weights as init) -----------
    model = HybridModel(
        h_rec=mcfg["h_rec"], K=mcfg["K"], transformer_config=qcfg,
        substrate=mcfg.get("substrate"), margin_h=mcfg["margin_h"],
        margin_c=mcfg["margin_c"], freeze_backbone=mcfg["freeze_backbone"],
        use_embedding_input=mcfg.get("use_embedding_input", True),
        dtype=dtype,
    ).to(device)
    model.to(dtype=dtype)        # cast everything to compute dtype...
    model.recursion.float()      # ...except Recursion (fp32 certificate; see note)
    # γ MUST be an fp32 master parameter. model.to(dtype) above casts EVERY param
    # regardless of its declared dtype, so γ becomes bf16; a bf16 γ has ULP
    # 2^-12≈2.4e-4 near 2^-5 (> the 1e-4 AdamW step), snaps to 1/32 and freezes.
    # Re-cast here, BEFORE the optimizer is built, so AdamW allocates fp32 state
    # and γ updates freely. (Verified: γ sailed past 0.03125 on the V100 run.)
    model.transformer.gamma.data = model.transformer.gamma.data.float()
    info = model.transformer.load_pretrained(ckpt_dir)  # R1 init into the Aeon backbone
    print(f"[init] R1 weights loaded: {info['loaded']} tensors | "
          f"missing={len(info['missing'])} unexpected={len(info['unexpected'])}")
    if info["unexpected"]:
        print(f"  [WARN] unexpected keys (first few): {info['unexpected'][:5]}")

    # ---- data ------------------------------------------------------------
    from transformers import AutoTokenizer
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(ckpt_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    split = load_dataset(dcfg["dataset"], split="train")
    ds = AlpacaDataset(split, tok, dcfg["seq_len"])
    loader = torch.utils.data.DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=True)

    # ---- optimizer -------------------------------------------------------
    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=tcfg.get("weight_decay", 0.0))
    n_train = sum(p.numel() for p in params)
    print(f"[init] trainable params: {n_train/1e6:.2f}M | device={device} dtype={dtype}")
    print(f"[init] audit @ start: {model.audit()}")

    # ---- resume ----------------------------------------------------------
    start_step = 0
    if tcfg.get("resume"):
        ck = latest_checkpoint(tcfg["out_dir"])
        if ck:
            blob = torch.load(ck, map_location=device)
            model.load_state_dict(blob["model"], strict=False)
            opt.load_state_dict(blob["optim"])
            start_step = blob["step"]
            print(f"[resume] from {ck} at step {start_step}")

    # ---- train loop ------------------------------------------------------
    model.train()
    step = start_step
    t0 = time.time()
    done = False
    while not done:
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"])
            loss = out.loss if hasattr(out, "loss") else out["loss"]
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if tcfg.get("grad_clip"):
                torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
            opt.step()
            step += 1

            if step % tcfg["log_every"] == 0:
                a = model.audit()
                dt = time.time() - t0
                print(f"[step {step}] loss={loss.item():.4f} "
                      f"sigma_Wh={a['sigma_Wh']:.4f} sigma_Wc={a['sigma_Wc']:.4f} "
                      f"holds={a['holds']} lambda={a['lambda']:.3f} gamma={a['gamma']:.4e} "
                      f"({dt:.1f}s)")
                if not a["holds"]:
                    print("  [WARN] sigma certificate does NOT hold — investigate")

            if step % tcfg["ckpt_every"] == 0:
                p = save_checkpoint(tcfg["out_dir"], step, model, opt)
                print(f"[ckpt] {p}")

            if step >= tcfg["max_steps"]:
                done = True
                break

    save_checkpoint(tcfg["out_dir"], step, model, opt)
    print(f"[done] final step {step}")


if __name__ == "__main__":
    main()
