"""scripts/colab/evaluate_and_generate.py — evaluation + raw generations
after a major Aeon fluency checkpoint.

Modes:
  --mode stage1_valid    perplexity on WikiText-103 wiki.valid.raw
  --mode stage1_test     perplexity on WikiText-103 wiki.test.raw (LOCK USE:
                          only for final Stage-1 promotion, never as
                          checkpoint-selection signal)
  --mode stage2_val      response-masked loss on the locked stage2_val
                          partition (checkpoint-selection signal only —
                          NEVER used for promotion). Verifies
                          stage2_val_lock_sha256 before scoring.
  --mode stage2_fresh    response-masked loss on dolly15k_fresh_eval
                          (single-use campaign promotion gate; verifies
                          the fresh_eval_lock_sha256 before scoring;
                          refuses on drift)
  --mode generate        produce raw unedited generations from a prompt list
                          under the frozen AttributionSettings contract

Never calls another model. Never rewrites text. Streamed decode matches
one-shot decode (D_stream == D_full).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def _sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _load_ck(model, path: Path, device):
    import torch
    st = torch.load(str(path), map_location=device, weights_only=False)
    if isinstance(st, dict) and "model_state_dict" in st:
        state = st["model_state_dict"]
    else:
        state = st
    m, u = model.load_state_dict(state, strict=False)
    assert not m and not u, f"load: missing={m[:3]} unexpected={u[:3]}"


def _wikitext_perplexity(model, tok, path: Path, device,
                          seq_len: int = 512, batch_size: int = 4,
                          max_batches: int = 200):
    import torch
    import torch.nn.functional as F
    from scripts.colab.train_stage import _stream_wikitext_tokens

    model.eval()
    total_nll, total_tok = 0.0, 0
    buf = []
    batches_done = 0
    with torch.inference_mode():
        for tid in _stream_wikitext_tokens(tok, path):
            buf.append(tid)
            need = batch_size * (seq_len + 1)
            if len(buf) >= need:
                chunk = torch.tensor(buf[:need], dtype=torch.long,
                                       device=device).view(batch_size, seq_len + 1)
                buf = buf[need:]
                input_ids = chunk[:, :-1].contiguous()
                targets = chunk[:, 1:].contiguous()
                out = model(input_ids=input_ids)
                logits = out.logits.float()
                nll = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1), reduction="sum")
                total_nll += float(nll.item())
                total_tok += int(targets.numel())
                batches_done += 1
                if batches_done >= max_batches:
                    break
    if total_tok == 0:
        return float("nan"), 0, 0.0, 0
    avg_nll = total_nll / total_tok
    ppl = float(torch.exp(torch.tensor(avg_nll)).item())
    return avg_nll, total_tok, ppl, batches_done


def _stage2_fresh_eval(model, tok, root: Path, device,
                        seq_len: int = 256, batch_size: int = 4):
    """Response-masked loss on the fresh_eval subset. Verifies the
    fresh_eval_lock_sha256 before scoring; refuses on drift."""
    import torch
    from aeon.en_train.losses import conversational_loss, build_response_mask
    from aeon.en_train.proof_pilot import render_dolly_record_for_training

    fresh = json.loads((root / "docs/en_train/dolly15k_fresh_eval_manifest.json"
                        ).read_text(encoding="utf-8"))
    fresh_ids = sorted(fresh["fresh_eval"]["record_ids"])
    live_lock = _sha256_bytes("\n".join(fresh_ids).encode("utf-8"))
    if live_lock != fresh["fresh_eval"]["fresh_eval_lock_sha256"]:
        raise RuntimeError(
            f"fresh_eval_lock drift: recorded={fresh['fresh_eval']['fresh_eval_lock_sha256']} "
            f"recomputed={live_lock}")

    raw = {}
    src = root / "research-data/incoming/EN-DOLLY-15K/sources/databricks-dolly-15k.jsonl"
    for i, ln in enumerate(open(src)):
        if ln.strip():
            r = json.loads(ln)
            raw[f"dolly-{i:05d}"] = r

    model.eval()
    total_loss, total_tok = 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(fresh_ids), batch_size):
            batch = fresh_ids[i:i + batch_size]
            ids_batch, mask_batch = [], []
            for rid in batch:
                r = raw[rid]
                text, spans = render_dolly_record_for_training(
                    instruction=r.get("instruction", ""),
                    context=r.get("context", ""),
                    response=r.get("response", ""))
                ids, rmask = build_response_mask(tok, text, spans)
                ids = ids[:seq_len]; rmask = rmask[:seq_len]
                pad = seq_len - len(ids)
                ids += [0] * pad; rmask += [0] * pad
                ids_batch.append(ids); mask_batch.append(rmask)
            input_ids = torch.tensor(ids_batch, dtype=torch.long, device=device)
            resp_mask = torch.tensor(mask_batch, dtype=torch.long, device=device)
            attn = (input_ids != 0).long()
            if resp_mask.sum().item() == 0:
                continue
            loss, vt = conversational_loss(model, input_ids=input_ids,
                                             response_mask=resp_mask,
                                             attention_mask=attn)
            total_loss += float(loss.item()) * vt
            total_tok += int(vt)
    if total_tok == 0:
        return float("nan"), 0
    return total_loss / total_tok, total_tok


def _generate_greedy(model, tok, prompt_text: str, max_new_tokens: int,
                      device, max_context: int = 1024):
    """Greedy generation. Returns generated_ids."""
    import torch
    ids = list(tok.encode(prompt_text, add_bos=False, add_eos=False))
    if len(ids) > max_context - max_new_tokens:
        ids = ids[-(max_context - max_new_tokens):]
    generated = []
    stop = "max_new_tokens"
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            x = torch.tensor([ids + generated], dtype=torch.long, device=device)
            out = model(input_ids=x)
            nxt = int(out.logits[0, -1, :].argmax(dim=-1).item())
            generated.append(nxt)
            if nxt == tok.eos_id:
                stop = "eos"; break
    return generated, stop


def _stream_and_full(tok, ids):
    if not ids:
        return "", ""
    D_full = tok.decode(ids)
    emitted = ""
    deltas = []
    for i in range(1, len(ids) + 1):
        canonical_so_far = tok.decode(ids[:i])
        committable = canonical_so_far.rstrip("�")
        if committable.startswith(emitted):
            deltas.append(committable[len(emitted):])
            emitted = committable
    if emitted != D_full:
        tail = D_full[len(emitted):] if D_full.startswith(emitted) else D_full
        deltas.append(tail)
    return "".join(deltas), D_full


def _stage2_val_eval(model, tok, root: Path, device,
                      seq_len: int = 256, batch_size: int = 4):
    """Response-masked loss on the locked stage2_val partition.
    Verifies the stage2_val_lock_sha256 before scoring."""
    import torch
    from aeon.en_train.losses import conversational_loss, build_response_mask
    from aeon.en_train.proof_pilot import render_dolly_record_for_training

    fresh = json.loads((root / "docs/en_train/dolly15k_fresh_eval_manifest.json"
                        ).read_text(encoding="utf-8"))
    if "stage2_validation" not in fresh:
        raise RuntimeError("stage2_validation partition missing from manifest")
    ids = sorted(fresh["stage2_validation"]["record_ids"])
    live = _sha256_bytes("\n".join(ids).encode("utf-8"))
    if live != fresh["stage2_validation"]["stage2_val_lock_sha256"]:
        raise RuntimeError(
            f"stage2_val_lock drift: recorded="
            f"{fresh['stage2_validation']['stage2_val_lock_sha256']} "
            f"recomputed={live}")

    # And it must not overlap fresh_eval (belt+braces at eval time).
    if set(ids) & set(fresh["fresh_eval"]["record_ids"]):
        raise RuntimeError("stage2_val overlaps fresh_eval — refusing to score")

    raw = {}
    src = root / "research-data/incoming/EN-DOLLY-15K/sources/databricks-dolly-15k.jsonl"
    for i, ln in enumerate(open(src)):
        if ln.strip():
            r = json.loads(ln)
            raw[f"dolly-{i:05d}"] = r

    model.eval()
    total_loss, total_tok = 0.0, 0
    with torch.inference_mode():
        for i in range(0, len(ids), batch_size):
            batch = ids[i:i + batch_size]
            ids_batch, mask_batch = [], []
            for rid in batch:
                r = raw[rid]
                text, spans = render_dolly_record_for_training(
                    instruction=r.get("instruction", ""),
                    context=r.get("context", ""),
                    response=r.get("response", ""))
                sids, rmask = build_response_mask(tok, text, spans)
                sids = sids[:seq_len]; rmask = rmask[:seq_len]
                pad = seq_len - len(sids)
                sids += [0] * pad; rmask += [0] * pad
                ids_batch.append(sids); mask_batch.append(rmask)
            input_ids = torch.tensor(ids_batch, dtype=torch.long, device=device)
            resp_mask = torch.tensor(mask_batch, dtype=torch.long, device=device)
            attn = (input_ids != 0).long()
            if resp_mask.sum().item() == 0:
                continue
            loss, vt = conversational_loss(model, input_ids=input_ids,
                                             response_mask=resp_mask,
                                             attention_mask=attn)
            total_loss += float(loss.item()) * vt
            total_tok += int(vt)
    if total_tok == 0:
        return float("nan"), 0
    return total_loss / total_tok, total_tok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--mode", required=True,
                    choices=["stage1_valid", "stage1_test",
                             "stage2_val", "stage2_fresh", "generate"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--wikitext-root",
                    default="/content/wikitext-103-raw/wikitext-103-raw")
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-batches", type=int, default=200)
    ap.add_argument("--prompt-count", type=int, default=25,
                    help="generate mode: number of fresh_eval prompts to use")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    sys.path.insert(0, str(root))
    import torch
    if not torch.cuda.is_available():
        print("HALT: CUDA unavailable", file=sys.stderr); return 5
    device = torch.device("cuda")

    from scripts.colab.train_stage import _build_model_and_tokenizer
    model, tok = _build_model_and_tokenizer(root)
    model = model.to(device)
    _load_ck(model, Path(args.checkpoint), device)

    if args.mode in ("stage1_valid", "stage1_test"):
        fname = "wiki.valid.raw" if args.mode == "stage1_valid" else "wiki.test.raw"
        path = Path(args.wikitext_root) / fname
        avg_nll, tok_count, ppl, batches = _wikitext_perplexity(
            model, tok, path, device, max_batches=args.max_batches)
        result = {"mode": args.mode, "file": str(path),
                  "batches": batches, "tokens_scored": tok_count,
                  "avg_nll": avg_nll, "perplexity": ppl,
                  "checkpoint": args.checkpoint,
                  "note": ("stage1_test is for final Stage-1 promotion "
                            "ONLY; never use for checkpoint selection.")
                          if args.mode == "stage1_test" else None}
    elif args.mode == "stage2_val":
        loss, tok_count = _stage2_val_eval(model, tok, root, device)
        result = {"mode": "stage2_val", "response_masked_loss": loss,
                  "tokens_scored": tok_count, "checkpoint": args.checkpoint,
                  "note": ("checkpoint-selection signal only — NEVER a "
                            "promotion gate; that is fresh_eval's role.")}
    elif args.mode == "stage2_fresh":
        loss, tok_count = _stage2_fresh_eval(model, tok, root, device)
        result = {"mode": "stage2_fresh", "response_masked_loss": loss,
                  "tokens_scored": tok_count, "checkpoint": args.checkpoint}
    else:  # generate
        fresh = json.loads((root / "docs/en_train/dolly15k_fresh_eval_manifest.json"
                            ).read_text(encoding="utf-8"))
        prompt_ids = sorted(fresh["fresh_eval"]["record_ids"])[:args.prompt_count]
        raw = {}
        for i, ln in enumerate(open(root /
            "research-data/incoming/EN-DOLLY-15K/sources/databricks-dolly-15k.jsonl")):
            if ln.strip():
                r = json.loads(ln); raw[f"dolly-{i:05d}"] = r
        outs = []
        for pid in prompt_ids:
            r = raw[pid]
            user = r.get("instruction", "")
            if r.get("context"):
                user = user + "\n\n" + r["context"]
            prompt = f"user: {user}\n\nassistant: "
            t0 = time.time()
            gen_ids, stop = _generate_greedy(model, tok, prompt,
                                               args.max_new_tokens, device)
            streamed, full = _stream_and_full(tok, gen_ids)
            outs.append({
                "prompt_id": pid,
                "prompt_text": prompt,
                "prompt_category": r.get("category", ""),
                "generated_token_ids": gen_ids,
                "full_decoded_text": full,
                "streamed_decoded_text": streamed,
                "streamed_equals_full": streamed == full,
                "stop_reason": stop,
                "generation_duration_seconds": time.time() - t0,
                "checkpoint": args.checkpoint,
            })
        result = {"mode": "generate", "prompt_count": len(outs),
                  "raw_generations": outs,
                  "note": ("Raw, unedited generations. Renderer "
                            "streamed==full asserted per response. "
                            "Dylan review is the only approval gate.")}

    dump = json.dumps(result, indent=2, ensure_ascii=False)
    if args.out:
        Path(args.out).write_text(dump + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(dump)
    return 0


if __name__ == "__main__":
    sys.exit(main())
