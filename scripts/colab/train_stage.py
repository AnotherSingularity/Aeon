"""scripts/colab/train_stage.py — resumable native Aeon trainer.

Two stages:
  --stage stage1   general-English next-token training on WikiText-103 raw.
                   Parent = protected P2 (never overwritten).
  --stage stage2   response-masked instruction tuning on Dolly-15k (train
                   partition, excluding fresh_eval + retired IDs).
                   Parent = the best Stage-1 checkpoint.

Invariants preserved (fail-closed):
  * A0 architecture fingerprint unchanged
  * K = 16, MARGIN_H / MARGIN_C unchanged
  * parameter count / state_dict topology unchanged
  * tokenizer bytes unchanged
  * P2 bytes unchanged
  * theta immutable during inference (not called here — training only)

Resume:
  * Full torch.save({"model_state_dict", "optimizer_state_dict",
      "scheduler_state_dict", "torch_rng_state", "cuda_rng_state",
      "python_rng_state", "tokens_covered", "step", "config"}, path)
  * Checkpoint written every --checkpoint-every-tokens (default 250_000)
      AND every --checkpoint-every-seconds (default 1800 seconds).
  * On start, loads latest valid checkpoint from --checkpoint-dir if
      present; otherwise loads --parent.
  * Emits a training_log.jsonl per step and a checkpoint_index.json.

Never calls another model, API, teacher, judge, corrector, retrieval
system, LoRA, adapter, or fallback. Uses only aeon.hybrid.HybridModel
+ aeon.en_train.losses.masked_next_token_loss / conversational_loss.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple


def _sha256_file(p: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _load_pinned_hashes(root: Path) -> Dict[str, str]:
    fp = json.loads((root / "docs" / "en_train" /
                     "EN_TRAIN_ARCHITECTURE_FREEZE.json").read_text(encoding="utf-8"))
    return {
        "A0_digest": fp["architecture_fingerprint_A0_digest"],
        "P2_sha": fp["protected_p2_checkpoint"]["sha256"],
        "TOK_sha": fp["protected_tokenizer"]["sha256"],
        "total_parameters": fp["total_parameters"],
        "K": fp["K"],
    }


# ---------------------------------------------------------------------------
# Model + tokenizer builders
# ---------------------------------------------------------------------------
def _build_model_and_tokenizer(root: Path):
    import torch, yaml
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    from aeon.tokenizer import AeonTokenizer

    release_mf = json.load(open(root /
        "release-assets/aeon-desktop-p2-proxy/manifests/release_manifest.json"))
    arch_mf = json.load(open(root /
        "release-assets/aeon-desktop-p2-proxy/manifests/architecture_manifest.json"))
    model_cfg = yaml.safe_load(open(root /
        f"release-assets/aeon-desktop-p2-proxy/{arch_mf['config_relpath']}"))
    mcfg = model_cfg["model"]; tcfg = mcfg["transformer"]

    tconfig = AeonTransformerConfig(
        vocab_size=int(release_mf["tokenizer_vocab_size"]),
        hidden_size=tcfg["hidden_size"],
        num_hidden_layers=tcfg["num_hidden_layers"],
        num_attention_heads=tcfg["num_attention_heads"],
        num_key_value_heads=tcfg["num_key_value_heads"],
        head_dim=tcfg["head_dim"],
        intermediate_size=tcfg["intermediate_size"],
        max_position_embeddings=tcfg["max_position_embeddings"])
    model = HybridModel(transformer_config=tconfig, h_rec=mcfg["h_rec"],
                        K=int(mcfg["K"]), margin_h=mcfg["margin_h"],
                        margin_c=mcfg["margin_c"],
                        use_embedding_input=True, dtype=torch.float32
                       ).to(dtype=torch.float32)
    tok = AeonTokenizer(str(root /
        "release-assets/aeon-desktop-p2-proxy/tokenizer/aeon-lbc1.model"))
    return model, tok


# ---------------------------------------------------------------------------
# Stage-1 data (WikiText-103 raw): stream tokens from wiki.train.raw
# ---------------------------------------------------------------------------
def _iter_wikitext_lines(path: Path) -> Iterator[str]:
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.rstrip("\n")
            if s:
                yield s


def _stream_wikitext_tokens(tok, path: Path) -> Iterator[int]:
    """Yield token ids from a WikiText file, joining with newlines."""
    for line in _iter_wikitext_lines(path):
        ids = tok.encode(line, add_bos=False, add_eos=False)
        for i in ids:
            yield i


# ---------------------------------------------------------------------------
# Stage-2 data (Dolly): iterate untouched-train records excluding fresh_eval
# ---------------------------------------------------------------------------
def _load_dolly_stage2_exclusion_set(root: Path) -> set:
    """The set of Dolly record_ids that MUST NOT enter any Stage-2
    training batch. Belt+braces: EXPLICITLY includes:
      * fresh_eval  (single-use promotion gate for this campaign)
      * stage2_validation  (checkpoint-selection signal, never train)
      * pilot_consumed train (ENGLISH-PROOF-0 iterated these — retired)
      * retired val partition (ENGLISH-PROOF-0 checkpoint selection)
      * retired sealed_test partition (ENGLISH-PROOF-0 promotion signal)
    Anything in this set entering training is a leak; the caller's
    contract is 'training batches contain no member of this set'.
    """
    split = json.loads((root / "docs/en_train/dolly15k_split_manifest.json"
                        ).read_text(encoding="utf-8"))
    fresh = json.loads((root / "docs/en_train/dolly15k_fresh_eval_manifest.json"
                        ).read_text(encoding="utf-8"))
    excluded = set()
    excluded |= set(fresh["fresh_eval"]["record_ids"])
    excluded |= set(fresh["consumed_train_ids_by_pilot"])
    excluded |= set(fresh.get("stage2_validation", {}).get("record_ids", []))
    excluded |= set(split["val_ids"])
    excluded |= set(split["sealed_test_ids"])
    return excluded


def _load_dolly_stage2_pool(root: Path):
    """Return list of DollyRecord for Stage-2 training. Guaranteed
    disjoint from every set in _load_dolly_stage2_exclusion_set(root).
    A final in-function assertion fails loudly if any excluded id
    slips through."""
    from aeon.en_train.dolly_split import DollyRecord
    split = json.loads((root / "docs/en_train/dolly15k_split_manifest.json"
                        ).read_text(encoding="utf-8"))
    excluded = _load_dolly_stage2_exclusion_set(root)
    train_ids = [rid for rid in split["train_ids"] if rid not in excluded]

    src = root / "research-data/incoming/EN-DOLLY-15K/sources/databricks-dolly-15k.jsonl"
    raw = {}
    for i, ln in enumerate(open(src)):
        if ln.strip():
            r = json.loads(ln)
            raw[f"dolly-{i:05d}"] = r
    pool = []
    for rid in train_ids:
        r = raw.get(rid)
        if not r:
            continue
        pool.append(DollyRecord(record_id=rid,
                                 instruction=r.get("instruction", "") or "",
                                 context=r.get("context", "") or "",
                                 response=r.get("response", "") or "",
                                 category=r.get("category", "") or ""))
    # Final leak check: no pool record's id can be in the exclusion set.
    leaked = [p.record_id for p in pool if p.record_id in excluded]
    assert not leaked, f"Stage-2 training pool leaked excluded ids: {leaked[:5]}"
    return pool


def _load_dolly_stage2_validation(root: Path):
    """Return list of DollyRecord for the in-loop Stage-2 validation
    signal (checkpoint-selection). Locked at manifest time; the
    stage2_val_lock_sha256 is re-verified before returning."""
    import hashlib
    from aeon.en_train.dolly_split import DollyRecord
    fresh = json.loads((root / "docs/en_train/dolly15k_fresh_eval_manifest.json"
                        ).read_text(encoding="utf-8"))
    if "stage2_validation" not in fresh:
        return []
    ids = sorted(fresh["stage2_validation"]["record_ids"])
    canon = "\n".join(ids).encode("utf-8")
    got = "sha256:" + hashlib.sha256(canon).hexdigest()
    want = fresh["stage2_validation"]["stage2_val_lock_sha256"]
    if got != want:
        raise RuntimeError(
            f"stage2_val_lock drift: want={want} got={got}")

    # Also assert disjointness at load time.
    fresh_eval_ids = set(fresh["fresh_eval"]["record_ids"])
    pilot_consumed = set(fresh["consumed_train_ids_by_pilot"])
    for rid in ids:
        assert rid not in fresh_eval_ids, \
            f"stage2_val id {rid} overlaps fresh_eval (data corruption)"
        assert rid not in pilot_consumed, \
            f"stage2_val id {rid} overlaps pilot_consumed"

    src = root / "research-data/incoming/EN-DOLLY-15K/sources/databricks-dolly-15k.jsonl"
    raw = {}
    for i, ln in enumerate(open(src)):
        if ln.strip():
            r = json.loads(ln)
            raw[f"dolly-{i:05d}"] = r
    return [DollyRecord(record_id=rid,
                         instruction=raw[rid].get("instruction", "") or "",
                         context=raw[rid].get("context", "") or "",
                         response=raw[rid].get("response", "") or "",
                         category=raw[rid].get("category", "") or "")
             for rid in ids if rid in raw]


# ---------------------------------------------------------------------------
# Checkpoint IO
# ---------------------------------------------------------------------------
def _list_checkpoints(ck_dir: Path) -> List[Path]:
    if not ck_dir.exists():
        return []
    cks = sorted(ck_dir.glob("checkpoint_*.pt"))
    return cks


def _save_checkpoint(ck_dir: Path, *, model, optimizer, scheduler,
                      tokens_covered: int, step: int, stage: str,
                      config: Dict) -> Path:
    import torch
    ck_dir.mkdir(parents=True, exist_ok=True)
    tmp = ck_dir / f"checkpoint_{step:09d}.pt.tmp"
    final = ck_dir / f"checkpoint_{step:09d}.pt"
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available() else None),
        "python_rng_state": random.getstate(),
        "tokens_covered": int(tokens_covered),
        "step": int(step),
        "stage": stage,
        "config": config,
    }
    torch.save(state, tmp)
    os.replace(tmp, final)   # atomic rename
    # Update index
    idx = ck_dir / "checkpoint_index.json"
    idx_data = {"latest": str(final.name), "step": step,
                "tokens_covered": tokens_covered, "stage": stage,
                "saved_at_utc_epoch": time.time()}
    idx.write_text(json.dumps(idx_data, indent=2, sort_keys=True), encoding="utf-8")
    return final


def _load_latest_or_parent(ck_dir: Path, parent_ckpt: Path,
                            *, model, optimizer, scheduler, device,
                            is_p2_nested: bool = True):
    """Load the latest checkpoint from ck_dir if any; else load parent.
    Returns (loaded_from, tokens_covered, step)."""
    import torch
    cks = _list_checkpoints(ck_dir)
    if cks:
        latest = cks[-1]
        print(f"[train] resuming from {latest}")
        st = torch.load(str(latest), map_location=device, weights_only=False)
        model.load_state_dict(st["model_state_dict"], strict=False)
        try:
            optimizer.load_state_dict(st["optimizer_state_dict"])
        except Exception as e:
            print(f"[train] warning: optimizer resume failed: {e}")
        if scheduler is not None and st.get("scheduler_state_dict"):
            try:
                scheduler.load_state_dict(st["scheduler_state_dict"])
            except Exception as e:
                print(f"[train] warning: scheduler resume failed: {e}")
        try:
            torch.set_rng_state(st["torch_rng_state"])
            if torch.cuda.is_available() and st.get("cuda_rng_state"):
                torch.cuda.set_rng_state_all(st["cuda_rng_state"])
            random.setstate(st["python_rng_state"])
        except Exception as e:
            print(f"[train] warning: rng resume failed: {e}")
        return str(latest), int(st.get("tokens_covered", 0)), int(st.get("step", 0))
    # Otherwise fresh from parent
    print(f"[train] fresh start from {parent_ckpt}")
    st = torch.load(str(parent_ckpt), map_location=device, weights_only=True)
    if is_p2_nested and isinstance(st, dict) and "model_state_dict" in st:
        st = st["model_state_dict"]
    m, u = model.load_state_dict(st, strict=False)
    assert not m and not u, f"parent load: missing={m[:3]} unexpected={u[:3]}"
    return str(parent_ckpt), 0, 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".",
                    help="repo/bundle root")
    ap.add_argument("--stage", required=True, choices=["stage1", "stage2"])
    ap.add_argument("--parent", required=True,
                    help="parent checkpoint path (P2 for stage1, best stage1 for stage2)")
    ap.add_argument("--checkpoint-dir", required=True,
                    help="directory (e.g. on Google Drive) to save checkpoints in")
    ap.add_argument("--target-tokens", type=int, required=True,
                    help="halt after this many tokens covered")

    # Stage-1 data
    ap.add_argument("--wikitext-root", default="/content/wikitext-103-raw/wikitext-103-raw")
    # Stage-2 data
    ap.add_argument("--dolly-jsonl", default="research-data/incoming/EN-DOLLY-15K/sources/databricks-dolly-15k.jsonl")

    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup-steps", type=int, default=200)
    ap.add_argument("--grad-clip", type=float, default=1.0)

    ap.add_argument("--checkpoint-every-tokens", type=int, default=250_000)
    ap.add_argument("--checkpoint-every-seconds", type=int, default=1800)

    ap.add_argument("--val-every-steps", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=8)

    ap.add_argument("--dry-run", action="store_true",
                    help="verify pipeline, run 5 steps, save one checkpoint, exit")
    ap.add_argument("--wall-time-cap-seconds", type=int, default=0,
                    help="if >0, halt training after this many seconds; "
                         "used by Colab operator to cap a session cleanly")

    args = ap.parse_args()

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    import torch

    # -------- Sanity: invariance pins ----------
    pinned = _load_pinned_hashes(root)
    if pinned["K"] != 16:
        print(f"HALT: pinned K != 16 ({pinned['K']})", file=sys.stderr); return 3

    # -------- Device ----------
    if not torch.cuda.is_available():
        print("HALT: CUDA is unavailable. free-Colab GPU runtime required.",
              file=sys.stderr)
        return 5
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    # -------- Model + tokenizer ----------
    model, tok = _build_model_and_tokenizer(root)
    model = model.to(device)
    if tok.vocab_size != 16000:
        print(f"HALT: tokenizer vocab_size={tok.vocab_size} != 16000",
              file=sys.stderr); return 3

    # -------- Optimizer + scheduler ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    betas=(0.9, 0.95), weight_decay=0.0)
    # linear warmup then constant — simplest resumable choice.
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(1, args.warmup_steps)))

    # -------- Resume or load parent ----------
    ck_dir = Path(args.checkpoint_dir).resolve()
    parent = Path(args.parent).resolve()
    is_p2 = str(parent).endswith("aeon_lbc1_P2/final.pt")
    src_used, tokens_covered, step = _load_latest_or_parent(
        ck_dir, parent, model=model, optimizer=optimizer,
        scheduler=scheduler, device=device, is_p2_nested=is_p2)

    # -------- Freeze-time architecture check ----------
    from aeon.en_train.proof import (
        compute_architecture_fingerprint, digest_fingerprint,
        assert_architecture_invariant, sigma_certificate, check_finite_state_dict,
    )
    fp = compute_architecture_fingerprint(model)
    d = digest_fingerprint(fp)
    if d != pinned["A0_digest"]:
        print(f"HALT: A0 digest drift {d} != pinned {pinned['A0_digest']}",
              file=sys.stderr); return 3
    if fp["total_parameters"] != pinned["total_parameters"]:
        print(f"HALT: parameter count drift {fp['total_parameters']} != {pinned['total_parameters']}",
              file=sys.stderr); return 3
    sc = sigma_certificate(model)
    if abs(sc["MARGIN_H"] - 0.02) > 1e-9 or abs(sc["MARGIN_C"] - 0.02) > 1e-9:
        print(f"HALT: margin drift {sc}", file=sys.stderr); return 3
    check_finite_state_dict(model)

    # -------- Training log ----------
    ck_dir.mkdir(parents=True, exist_ok=True)
    log_path = ck_dir / f"training_log_{args.stage}.jsonl"
    log_fh = log_path.open("a", encoding="utf-8")

    config_dump = {
        "stage": args.stage, "parent": str(parent), "seq_len": args.seq_len,
        "batch_size": args.batch_size, "lr": args.lr,
        "target_tokens": args.target_tokens,
    }

    # -------- Batch iterator per stage ----------
    if args.stage == "stage1":
        wik_train = Path(args.wikitext_root) / "wiki.train.raw"
        if not wik_train.exists():
            print(f"HALT: wikitext train missing: {wik_train}", file=sys.stderr); return 4
        def stage1_batches():
            buf: List[int] = []
            while True:
                # Re-iterate the train file forever (epochs)
                for tid in _stream_wikitext_tokens(tok, wik_train):
                    buf.append(tid)
                    need = args.batch_size * (args.seq_len + 1)
                    if len(buf) >= need:
                        chunk = buf[:need]
                        buf = buf[need:]
                        import numpy as _np
                        arr = torch.tensor(chunk[:args.batch_size * (args.seq_len + 1)],
                                             dtype=torch.long)
                        arr = arr.view(args.batch_size, args.seq_len + 1)
                        input_ids = arr[:, :-1].contiguous()
                        targets = arr[:, 1:].contiguous()
                        yield input_ids, targets, None  # None -> full LM mask
        batch_iter = stage1_batches()

    else:  # stage2
        from aeon.en_train.proof_pilot import render_dolly_record_for_training
        from aeon.en_train.losses import build_response_mask
        pool = _load_dolly_stage2_pool(root)
        print(f"[train] stage2 pool size (excludes fresh_eval + retired): {len(pool)}")
        pool_ids = list(pool)
        rng_local = random.Random(20260824)
        rng_local.shuffle(pool_ids)
        def stage2_batches():
            cursor = 0; ordered = list(pool_ids)
            while True:
                if cursor + args.batch_size > len(ordered):
                    rng_local.shuffle(ordered); cursor = 0
                batch = ordered[cursor:cursor + args.batch_size]
                cursor += args.batch_size
                ids_batch, mask_batch = [], []
                for rec in batch:
                    text, spans = render_dolly_record_for_training(
                        instruction=rec.instruction, context=rec.context,
                        response=rec.response)
                    ids, rmask = build_response_mask(tok, text, spans)
                    ids = ids[:args.seq_len]; rmask = rmask[:args.seq_len]
                    pad = args.seq_len - len(ids)
                    ids += [0] * pad; rmask += [0] * pad
                    ids_batch.append(ids); mask_batch.append(rmask)
                input_ids = torch.tensor(ids_batch, dtype=torch.long)
                resp_mask = torch.tensor(mask_batch, dtype=torch.long)
                # attention = non-pad
                attn = (input_ids != 0).long()
                # For response-masked loss we use targets = shift + response_mask
                yield input_ids, resp_mask, attn
        batch_iter = stage2_batches()

    # -------- Loss helpers ----------
    from aeon.en_train.losses import masked_next_token_loss, conversational_loss

    # -------- Loop ----------
    start_ts = time.time()
    last_ck_ts = start_ts
    last_ck_tokens = tokens_covered
    if args.dry_run:
        max_steps = 5
    else:
        max_steps = 10**12
    model.train()
    print(f"[train] starting stage={args.stage} tokens_covered={tokens_covered} step={step}")
    while tokens_covered < args.target_tokens and step < max_steps:
        if args.wall_time_cap_seconds > 0 and (time.time() - start_ts) > args.wall_time_cap_seconds:
            print(f"[train] wall-time cap reached ({args.wall_time_cap_seconds}s); halting cleanly")
            break

        input_ids, mask, attn = next(batch_iter)
        input_ids = input_ids.to(device, non_blocking=True)
        if attn is not None:
            attn = attn.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if args.stage == "stage1":
            targets = mask.to(device, non_blocking=True)   # mask carries targets here
            out = model(input_ids=input_ids)
            # shift-based CE over the full sequence (no padding in stage1)
            logits = out.logits[:, :-1, :]
            tgt = targets[:, 1:] if targets.dim() == 2 and targets.size(1) == input_ids.size(1) else targets
            # Simplified stage-1: use masked_next_token_loss with mask of 1
            import torch.nn.functional as F
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                targets.reshape(-1))
            vt = int(targets.numel())
        else:
            resp_mask = mask.to(device, non_blocking=True)
            loss, vt = conversational_loss(model, input_ids=input_ids,
                                            response_mask=resp_mask,
                                            attention_mask=attn)
        if not torch.isfinite(loss).item():
            print(f"HALT: non-finite loss={loss.item()}", file=sys.stderr)
            return 6

        loss.backward()
        # Grad clip
        total_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_sq += float(p.grad.detach().to(torch.float32).pow(2).sum().item())
        gnorm = total_sq ** 0.5
        if args.grad_clip > 0 and gnorm > args.grad_clip:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(args.grad_clip / (gnorm + 1e-12))
        optimizer.step()
        scheduler.step()

        tokens_covered += int(vt)
        step += 1
        log_fh.write(json.dumps({
            "step": step, "stage": args.stage, "loss": float(loss.item()),
            "grad_l2": float(gnorm), "valid_tokens": int(vt),
            "tokens_covered": tokens_covered,
            "wall_seconds": time.time() - start_ts,
        }) + "\n"); log_fh.flush()

        if step % 25 == 0:
            print(f"[train] step={step} loss={loss.item():.3f} grad={gnorm:.2f} "
                  f"covered={tokens_covered} elapsed={time.time()-start_ts:.0f}s")

        # Checkpoint by tokens OR seconds
        by_tokens = (tokens_covered - last_ck_tokens) >= args.checkpoint_every_tokens
        by_time = (time.time() - last_ck_ts) >= args.checkpoint_every_seconds
        if by_tokens or by_time or args.dry_run:
            path = _save_checkpoint(ck_dir, model=model, optimizer=optimizer,
                                     scheduler=scheduler,
                                     tokens_covered=tokens_covered, step=step,
                                     stage=args.stage, config=config_dump)
            print(f"[train] checkpoint saved: {path}")
            last_ck_ts = time.time()
            last_ck_tokens = tokens_covered

    # Final save
    _save_checkpoint(ck_dir, model=model, optimizer=optimizer,
                     scheduler=scheduler,
                     tokens_covered=tokens_covered, step=step,
                     stage=args.stage, config=config_dump)
    print(f"[train] DONE stage={args.stage} tokens={tokens_covered} step={step}")

    # Post-training invariance re-check
    fp2 = compute_architecture_fingerprint(model)
    d2 = digest_fingerprint(fp2)
    if d2 != pinned["A0_digest"]:
        print(f"POST-HALT: A0 digest drift {d2}", file=sys.stderr); return 3
    assert_architecture_invariant(model)
    print("[train] post-training invariance OK")

    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
