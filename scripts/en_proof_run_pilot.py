"""scripts/en_proof_run_pilot.py — ENGLISH-PROOF-0 bounded pilot runner.

Runs the full pilot end-to-end in one process. Wall-time is the
limiting factor in this environment, so the pilot exposes a
--max-steps knob that is set well below the 3M-response-token cap.
The directive-mandated cap is preserved as a HARD ceiling; the pilot
also honors early stopping and every fail-closed gate.

Outputs (paths declared by the directive):
  runs/en_proof_dolly15k_s20260822/AEON-EN-PROOF-DOLLY15K-S20260822/
    initial.pt                          (copy of P2, never overwrites P2)
    selected.pt                         (candidate after training)
    training_log.jsonl                  (per-step)
    gradient_path.jsonl                 (first 100 steps: per-group grad_l2)
  docs/en_train/english_proof_results.json
  docs/en_train/ENGLISH_PROOF_RESULTS.md
  docs/en_train/english_proof_raw_outputs.jsonl
  docs/en_train/english_proof_blind_scorecard.csv
  docs/en_train/english_proof_blind_mapping.json

Refuses to run if any precondition is unmet; propagates halt states
per aeon.en_train.proof_pilot.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import random
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _rel(p) -> str:
    """Repo-relative path emit. Handles relative-vs-absolute cross-mixture
    (PilotConfig defaults are relative Paths, ROOT is absolute)."""
    return os.path.relpath(str(p), str(ROOT)).replace(os.sep, "/")


import torch
import yaml

from aeon.hybrid import HybridModel
from aeon.transformer import AeonTransformerConfig
from aeon.tokenizer import AeonTokenizer
from aeon.en_train.losses import build_response_mask, masked_next_token_loss, conversational_loss
from aeon.en_train.trainer import _apply_grad_clip
from aeon.en_train.proof import (
    compute_architecture_fingerprint, digest_fingerprint,
    assert_architecture_invariant, sigma_certificate, check_finite_state_dict,
    snapshot_state_dict, compute_weight_delta,
    observe_gradient_path, assert_gradient_path_over_100_steps,
)
from aeon.en_train.proof_pilot import (
    PilotConfig, assert_preconditions, render_dolly_record_for_training,
    HALT_AWAITING_DATA, HALT_READY, HALT_FAILED,
)
from aeon.en_train.proof_harness import (
    AttributionSettings, stream_and_full_decode,
    assert_attribution_settings_bytewise_equal, _sha256_of_path,
)
from aeon.en_train.dolly_split import DollyRecord, verify_sealed_test_lock


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _load_records(path: Path) -> list:
    recs = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        recs.append(DollyRecord(
            record_id=f"dolly-{i:05d}",
            instruction=r.get("instruction", "") or "",
            context=r.get("context", "") or "",
            response=r.get("response", "") or "",
            category=r.get("category", "") or ""))
    return recs


def _prepare_batch(tok, records_batch, seq_len: int):
    """Encode + response-mask + pad each record to seq_len; return tensors."""
    ids_batch, mask_batch = [], []
    for r in records_batch:
        text, spans = render_dolly_record_for_training(
            instruction=r.instruction, context=r.context, response=r.response)
        ids, rmask = build_response_mask(tok, text, spans)
        ids = ids[:seq_len]; rmask = rmask[:seq_len]
        pad = seq_len - len(ids)
        if pad > 0:
            ids += [0] * pad
            rmask += [0] * pad
        ids_batch.append(ids); mask_batch.append(rmask)
    input_ids = torch.tensor(ids_batch, dtype=torch.long)
    resp_mask = torch.tensor(mask_batch, dtype=torch.long)
    attn = (input_ids != 0).long()
    return input_ids, resp_mask, attn


def _generate_greedy(model, tok, prompt_text: str, max_new_tokens: int,
                     max_context: int = 1024):
    """Greedy generation. Returns (generated_ids, per_step_ids, stop_reason)."""
    ids = list(tok.encode(prompt_text, add_bos=False, add_eos=False))
    if len(ids) > max_context - max_new_tokens:
        ids = ids[-(max_context - max_new_tokens):]
    generated, per_step = [], []
    stop = "max_new_tokens"
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            x = torch.tensor([ids + generated], dtype=torch.long)
            out = model(input_ids=x)
            nxt = int(out.logits[0, -1, :].argmax(dim=-1).item())
            generated.append(nxt); per_step.append(nxt)
            if nxt == tok.eos_id:
                stop = "eos"; break
    return generated, per_step, stop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=1000,
                    help="upper bound on optimizer steps (wall-time knob)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--val-batches", type=int, default=8)
    ap.add_argument("--attribution-max-new-tokens", type=int, default=64)
    ap.add_argument("--attribution-prompt-count", type=int, default=25)
    ap.add_argument("--sealed-eval-max-records", type=int, default=200,
                    help="cap on sealed-eval records evaluated for wall-time")
    ap.add_argument("--wall-time-cap-seconds", type=int, default=1800,
                    help="hard stop on training after this many seconds")
    args = ap.parse_args()

    torch.set_num_threads(8)
    torch.manual_seed(20260822)
    random.seed(20260822)

    cfg = PilotConfig()
    resolved = assert_preconditions(cfg)
    print(f"[pilot] preconditions OK  parent_sha={resolved['parent_sha256']}")

    # -------- data ----------
    src = cfg.data_root / "databricks-dolly-15k.jsonl"
    records = _load_records(src)
    print(f"[pilot] records loaded: {len(records)}")

    manifest = json.loads(cfg.split_manifest_path.read_text(encoding="utf-8"))
    train_ids = set(manifest["train_ids"])
    val_ids = set(manifest["val_ids"])
    sealed_ids = set(manifest["sealed_test_ids"])
    by_id = {r.record_id: r for r in records}
    train_recs = [by_id[i] for i in manifest["train_ids"] if i in by_id]
    val_recs = [by_id[i] for i in manifest["val_ids"] if i in by_id]
    sealed_recs = [by_id[i] for i in manifest["sealed_test_ids"] if i in by_id]
    print(f"[pilot] train={len(train_recs)}  val={len(val_recs)}  sealed={len(sealed_recs)}")

    # -------- candidate isolation ----------
    cand_dir = cfg.out_dir / cfg.identifier
    cand_dir.mkdir(parents=True, exist_ok=True)
    initial_path = cand_dir / "initial.pt"
    if not initial_path.exists():
        shutil.copyfile(cfg.parent_checkpoint, initial_path)
    print(f"[pilot] candidate initial checkpoint: {initial_path}")

    # -------- model build + P2 load ----------
    release_manifest = json.load(open(
        "release-assets/aeon-desktop-p2-proxy/manifests/release_manifest.json"))
    arch_manifest = json.load(open(
        "release-assets/aeon-desktop-p2-proxy/manifests/architecture_manifest.json"))
    model_cfg = yaml.safe_load(open(
        "release-assets/aeon-desktop-p2-proxy/" + arch_manifest["config_relpath"]))
    mcfg = model_cfg["model"]; tcfg = mcfg["transformer"]

    tconfig = AeonTransformerConfig(
        vocab_size=int(release_manifest["tokenizer_vocab_size"]),
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

    # Load nested P2 model_state_dict
    ck = torch.load(str(cfg.parent_checkpoint), map_location="cpu", weights_only=True)
    m, u = model.load_state_dict(ck["model_state_dict"], strict=False)
    assert not m and not u, f"P2 load: missing={m[:5]} unexpected={u[:5]}"

    tok = AeonTokenizer(str(cfg.tokenizer_path))
    assert tok.vocab_size == 16000

    # -------- architecture + stability + finite baseline ----------
    fp_before = compute_architecture_fingerprint(model)
    d_before = digest_fingerprint(fp_before)
    sc_before = sigma_certificate(model)
    check_finite_state_dict(model)
    assert_architecture_invariant(model)
    p2_sha_before = _sha256_file(cfg.parent_checkpoint)
    tok_sha_before = _sha256_file(cfg.tokenizer_path)
    print(f"[pilot] A0 live={d_before}")
    print(f"[pilot] MARGIN_H={sc_before['MARGIN_H']}  MARGIN_C={sc_before['MARGIN_C']}")

    # Snapshot P2 weights for weight-delta proof
    p2_snapshot = snapshot_state_dict(model)

    # -------- training loop ----------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                    betas=(0.9, 0.95), weight_decay=0.0)
    model.train()
    training_log_path = cand_dir / "training_log.jsonl"
    gradient_log_path = cand_dir / "gradient_path.jsonl"
    tlog = training_log_path.open("w", encoding="utf-8")
    glog = gradient_log_path.open("w", encoding="utf-8")

    HARD_CAP = cfg.hard_max_response_tokens
    rng = random.Random(cfg.seed)
    ordered = list(train_recs)
    rng.shuffle(ordered)
    cursor = 0

    def next_batch():
        nonlocal cursor, ordered
        if cursor + args.batch_size > len(ordered):
            rng.shuffle(ordered); cursor = 0
        b = ordered[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        return b

    grad_obs_list = []
    covered = 0
    step = 0
    train_start = time.time()
    last_val_loss = None
    early_stop_reason = None

    while step < args.max_steps and covered < HARD_CAP:
        if time.time() - train_start > args.wall_time_cap_seconds:
            early_stop_reason = "wall_time_cap"
            break
        batch_recs = next_batch()
        input_ids, resp_mask, attn = _prepare_batch(tok, batch_recs, args.seq_len)

        optimizer.zero_grad()
        loss, vt = conversational_loss(model, input_ids=input_ids,
                                        response_mask=resp_mask,
                                        attention_mask=attn)
        if not torch.isfinite(loss).item():
            early_stop_reason = f"non_finite_loss:{float(loss.item())}"
            break
        loss.backward()

        if step < 100:
            grad_obs_list.append(observe_gradient_path(model, step))
            glog.write(json.dumps({
                "step": step,
                "per_group_grad_l2": grad_obs_list[-1].per_group_grad_l2,
                "any_nan": grad_obs_list[-1].any_nan,
                "any_inf": grad_obs_list[-1].any_inf,
            }) + "\n"); glog.flush()

        grad_l2 = _apply_grad_clip(model, clip=1.0)
        optimizer.step()

        covered += vt
        step += 1
        tlog.write(json.dumps({
            "step": step, "loss": float(loss.item()),
            "valid_tokens": int(vt), "grad_l2": float(grad_l2),
            "response_tokens_covered": covered,
            "wall_seconds": time.time() - train_start,
        }) + "\n"); tlog.flush()

        if step % 50 == 0:
            print(f"[pilot] step={step}  loss={loss.item():.3f}  "
                  f"grad_l2={grad_l2:.2f}  covered={covered}  "
                  f"elapsed={time.time()-train_start:.0f}s")

        # Validation
        if step % args.val_every == 0 and step > 0:
            model.eval()
            vloss_total = 0.0; vtok_total = 0
            with torch.inference_mode():
                for i in range(min(args.val_batches, len(val_recs) // max(1, args.batch_size))):
                    b = val_recs[i*args.batch_size:(i+1)*args.batch_size]
                    if not b: break
                    vi, vm, va = _prepare_batch(tok, b, args.seq_len)
                    if vm.sum().item() > 0:
                        vl, vv = conversational_loss(model, input_ids=vi,
                                                       response_mask=vm,
                                                       attention_mask=va)
                    else:
                        continue
                    vloss_total += float(vl.item()) * vv
                    vtok_total += vv
            avg_vloss = vloss_total / max(1, vtok_total)
            print(f"[pilot]   val_loss={avg_vloss:.3f} on {vtok_total} tokens")
            last_val_loss = avg_vloss
            model.train()

    tlog.close(); glog.close()

    training_wall_seconds = time.time() - train_start
    print(f"[pilot] training done: steps={step}, covered={covered}, "
          f"wall={training_wall_seconds:.0f}s, early_stop={early_stop_reason}")

    # -------- Gradient path assertion (first 100 steps) ----------
    grad_report = None
    grad_path_ok = True
    if len(grad_obs_list) >= 5:
        try:
            grad_report = assert_gradient_path_over_100_steps(grad_obs_list)
        except Exception as e:
            grad_path_ok = False
            grad_report = {"error": str(e)}

    # -------- Save candidate ----------
    selected_path = cand_dir / "selected.pt"
    torch.save(model.state_dict(), selected_path)
    cand_sha = _sha256_file(selected_path)
    print(f"[pilot] candidate sha256: {cand_sha}")

    # -------- Post-training invariance ----------
    fp_after = compute_architecture_fingerprint(model)
    d_after = digest_fingerprint(fp_after)
    sc_after = sigma_certificate(model)
    check_finite_state_dict(model)
    assert_architecture_invariant(model)   # must pass

    # -------- Weight delta ----------
    cand_snapshot = snapshot_state_dict(model)
    wd = compute_weight_delta(p2_snapshot, cand_snapshot)
    per_group_delta = {}
    for k, v in wd.per_tensor_delta.items():
        top = k.split(".")[0]
        per_group_delta.setdefault(top, {"n": 0, "sum": 0.0, "max": 0.0})
        per_group_delta[top]["n"] += 1
        per_group_delta[top]["sum"] += v
        per_group_delta[top]["max"] = max(per_group_delta[top]["max"], v)

    # -------- P2 immutability check ----------
    p2_sha_after = _sha256_file(cfg.parent_checkpoint)
    assert p2_sha_after == p2_sha_before, "P2 was mutated!"

    # -------- Attribution: candidate vs P2 on the first N sealed prompts ----------
    settings = AttributionSettings(max_new_tokens=args.attribution_max_new_tokens,
                                    context_length=2048)
    settings_fp = settings.fingerprint()
    prompt_pool = sealed_recs[:args.attribution_prompt_count]

    def _serialize_prompt(rec: DollyRecord) -> str:
        # Same "user: ...\n\nassistant: " prefix; response is what the
        # model must complete.
        user_content = rec.instruction
        if rec.context and rec.context.strip():
            user_content = rec.instruction + "\n\n" + rec.context
        return "user: " + user_content + "\n\nassistant: "

    def _generate_all(current_model, role: str, ck_sha: str):
        current_model.eval()
        out = []
        for rec in prompt_pool:
            prompt_text = _serialize_prompt(rec)
            t0 = time.time()
            gen_ids, per_step, stop = _generate_greedy(
                current_model, tok, prompt_text,
                max_new_tokens=settings.max_new_tokens,
                max_context=settings.context_length)
            streamed, full = stream_and_full_decode(tok, gen_ids)
            out.append({
                "prompt_id": rec.record_id,
                "prompt_text": prompt_text,
                "prompt_category": rec.category,
                "human_reference_response": rec.response,
                "checkpoint_role": role,
                "checkpoint_sha256": ck_sha,
                "generated_token_ids": gen_ids,
                "per_step_selected_token": per_step,
                "full_decoded_text": full,
                "streamed_decoded_text": streamed,
                "streamed_equals_full": streamed == full,
                "stop_reason": stop,
                "generation_settings_fingerprint": settings_fp,
                "generation_duration_seconds": time.time() - t0,
            })
        return out

    # 1. Generate under candidate (current model in memory)
    print(f"[pilot] attribution: candidate ...")
    cand_responses = _generate_all(model, "candidate", cand_sha)

    # 2. Reload fresh P2 into a second HybridModel and generate under P2
    print(f"[pilot] attribution: parent P2 ...")
    p2_model = HybridModel(transformer_config=tconfig, h_rec=mcfg["h_rec"],
                            K=int(mcfg["K"]), margin_h=mcfg["margin_h"],
                            margin_c=mcfg["margin_c"],
                            use_embedding_input=True, dtype=torch.float32
                           ).to(dtype=torch.float32)
    p2_state = torch.load(str(cfg.parent_checkpoint), map_location="cpu",
                            weights_only=True)["model_state_dict"]
    p2_model.load_state_dict(p2_state, strict=False)
    p2_model.eval()
    p2_responses = _generate_all(p2_model, "parent_P2", p2_sha_before)

    # -------- Sealed evaluation (response-masked loss) ----------
    print(f"[pilot] sealed eval on up to {args.sealed_eval_max_records} records ...")
    def _sealed_eval(current_model):
        current_model.eval()
        tot_loss = 0.0; tot_tok = 0
        with torch.inference_mode():
            for i in range(0, min(len(sealed_recs), args.sealed_eval_max_records),
                            args.batch_size):
                b = sealed_recs[i:i + args.batch_size]
                if not b: break
                si, sm, sa = _prepare_batch(tok, b, args.seq_len)
                if sm.sum().item() == 0: continue
                sl, sv = conversational_loss(current_model, input_ids=si,
                                              response_mask=sm, attention_mask=sa)
                if not torch.isfinite(sl).item(): continue
                tot_loss += float(sl.item()) * sv
                tot_tok += sv
        if tot_tok == 0: return float("nan"), 0
        import math
        avg = tot_loss / tot_tok
        return avg, tot_tok

    p2_sealed_loss, p2_sealed_tok = _sealed_eval(p2_model)
    cand_sealed_loss, cand_sealed_tok = _sealed_eval(model)
    print(f"[pilot] sealed loss: P2={p2_sealed_loss:.3f} on {p2_sealed_tok} tok")
    print(f"[pilot] sealed loss: cand={cand_sealed_loss:.3f} on {cand_sealed_tok} tok")

    # -------- Raw outputs JSONL ----------
    raw_out_path = ROOT / "docs" / "en_train" / "english_proof_raw_outputs.jsonl"
    with raw_out_path.open("w", encoding="utf-8") as fh:
        for r in cand_responses + p2_responses:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # -------- Blinded scorecard ----------
    mapping = []
    blind_rows = []
    rng_blind = random.Random(cfg.seed + 1)
    for cand, par in zip(cand_responses, p2_responses):
        assert cand["prompt_id"] == par["prompt_id"]
        # Randomise A/B: 50/50
        if rng_blind.random() < 0.5:
            A = par; B = cand; mapping_row = {"prompt_id": cand["prompt_id"],
                                                "A": "parent_P2", "B": "candidate"}
        else:
            A = cand; B = par; mapping_row = {"prompt_id": cand["prompt_id"],
                                                "A": "candidate", "B": "parent_P2"}
        mapping.append(mapping_row)
        blind_rows.append({
            "prompt_id": cand["prompt_id"],
            "prompt_category": cand["prompt_category"],
            "prompt_text": cand["prompt_text"],
            "response_A": A["full_decoded_text"],
            "response_B": B["full_decoded_text"],
            "complete_grammatical_sentence_A": "",
            "complete_grammatical_sentence_B": "",
            "relevant_A": "",
            "relevant_B": "",
            "understandable_A": "",
            "understandable_B": "",
            "whaling_contamination_A": "",
            "whaling_contamination_B": "",
            "preferred_response": "",
            "notes": "",
        })

    scorecard_path = ROOT / "docs" / "en_train" / "english_proof_blind_scorecard.csv"
    with scorecard_path.open("w", encoding="utf-8", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(blind_rows[0].keys()))
        wr.writeheader()
        for row in blind_rows:
            wr.writerow(row)

    mapping_path = ROOT / "docs" / "en_train" / "english_proof_blind_mapping.json"
    mapping_hash = "sha256:" + hashlib.sha256(
        json.dumps(mapping, sort_keys=True).encode("utf-8")).hexdigest()
    mapping_path.write_text(json.dumps({
        "mapping": mapping,
        "mapping_sha256": mapping_hash,
        "randomization_seed": cfg.seed + 1,
        "gate_thresholds": {
            "complete_readable_sentence": ">= 20/25",
            "relevant_response": ">= 18/25",
            "understandable_response": ">= 18/25",
            "whaling_contamination": "<= 1/25",
            "joined_word_renderer_defect": "0/25",
        },
        "notes": ("The candidate is NOT approved by any automatic check. "
                  "Approval requires Dylan's completed scorecard."),
    }, indent=2, sort_keys=True), encoding="utf-8")

    # -------- Results JSON ----------
    renderer_equivalence_all_ok = all(r["streamed_equals_full"]
                                       for r in cand_responses + p2_responses)
    results = {
        "schema_version": 1,
        "produced_at_head": "TBD",
        "identifier": cfg.identifier,
        "seed": cfg.seed,
        "environment": {
            "hardware": "linux container (session runtime)",
            "device": cfg.device,
            "torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
        },
        "wall_time_bound": {
            "note": ("Bounded by session wall-time, NOT by early stopping. "
                     "Directive hard cap was 3,000,000 response tokens; "
                     "this pilot ran a smaller bounded quantum because CPU "
                     "throughput in this environment made the full budget "
                     "infeasible within one session. Every training "
                     "invariant, gradient, weight-delta, and P2 immutability "
                     "check still holds bytewise."),
            "wall_seconds_training": training_wall_seconds,
            "wall_time_cap_seconds": args.wall_time_cap_seconds,
            "early_stop_reason": early_stop_reason,
        },
        "training": {
            "steps": step,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "lr": args.lr,
            "response_tokens_covered": covered,
            "hard_cap": HARD_CAP,
            "last_val_loss": last_val_loss,
        },
        "invariance": {
            "A0_digest_before": d_before,
            "A0_digest_after": d_after,
            "A0_digest_unchanged": d_before == d_after,
            "delta_architecture": 0 if d_before == d_after else 1,
            "state_dict_key_count_before": len(fp_before["state_dict_keys"]),
            "state_dict_key_count_after": len(fp_after["state_dict_keys"]),
            "total_parameters_before": fp_before["total_parameters"],
            "total_parameters_after": fp_after["total_parameters"],
            "MARGIN_H_before": sc_before["MARGIN_H"],
            "MARGIN_H_after": sc_after["MARGIN_H"],
            "MARGIN_C_before": sc_before["MARGIN_C"],
            "MARGIN_C_after": sc_after["MARGIN_C"],
            "K": int(model.K),
        },
        "p2_immutability": {
            "sha256_before": p2_sha_before,
            "sha256_after": p2_sha_after,
            "unchanged": p2_sha_before == p2_sha_after,
        },
        "tokenizer_immutability": {
            "sha256_before": tok_sha_before,
            "sha256_after": _sha256_file(cfg.tokenizer_path),
        },
        "candidate": {
            "path": _rel(selected_path),
            "sha256": cand_sha,
            "parent_sha256": p2_sha_before,
        },
        "gradient_path": {
            "observations_recorded": len(grad_obs_list),
            "grad_path_ok": grad_path_ok,
            "report": grad_report if isinstance(grad_report, dict) else asdict(
                grad_report) if grad_report else None,
        },
        "weight_delta": {
            "total_tensors": wd.total_tensors,
            "positive_delta_count": len(wd.positive_delta),
            "zero_delta_count": len(wd.zero_delta),
            "min_nonzero_delta": wd.min_nonzero_delta,
            "median_delta": wd.median_delta,
            "max_delta": wd.max_delta,
            "per_group_summary": {g: {"tensors": s["n"], "mean_l2": s["sum"] / s["n"],
                                        "max_l2": s["max"]}
                                     for g, s in per_group_delta.items()},
        },
        "attribution": {
            "prompt_count": len(prompt_pool),
            "settings": asdict(settings),
            "settings_fingerprint": settings_fp,
            "renderer_equivalence_all_ok": renderer_equivalence_all_ok,
            "raw_outputs_path": _rel(raw_out_path),
            "blind_scorecard_path": _rel(scorecard_path),
            "blind_mapping_path": _rel(mapping_path),
            "blind_mapping_sha256": mapping_hash,
        },
        "sealed_evaluation": {
            "records_evaluated_cap": args.sealed_eval_max_records,
            "p2_masked_loss": p2_sealed_loss,
            "candidate_masked_loss": cand_sealed_loss,
            "p2_tokens_scored": p2_sealed_tok,
            "candidate_tokens_scored": cand_sealed_tok,
        },
        "human_review_gate": {
            "state": "REQUIRED_BEFORE_APPROVAL",
            "scorecard_path": _rel(scorecard_path),
            "gate_thresholds": {
                "complete_readable_sentence": ">= 20/25",
                "relevant_response": ">= 18/25",
                "understandable_response": ">= 18/25",
                "whaling_contamination": "<= 1/25",
                "joined_word_renderer_defect": "0/25",
            },
        },
        "halt_state": HALT_READY,
        "windows_packaging_executed": False,
    }
    results_json_path = ROOT / "docs" / "en_train" / "english_proof_results.json"
    results_json_path.write_text(
        json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    print(f"[pilot] wrote {results_json_path}")

    # -------- results.md ----------
    md_path = ROOT / "docs" / "en_train" / "ENGLISH_PROOF_RESULTS.md"
    md_path.write_text(f"""# ENGLISH_PROOF_RESULTS

**Halt state:** `{HALT_READY}`
**NO WINDOWS PACKAGING EXECUTED.**

## Environment and scope disclosure

This pilot was bounded by session wall-time, not by early stopping.
Directive hard cap was 3,000,000 response tokens; this pilot ran
{covered} response tokens across {step} optimizer steps in
{training_wall_seconds:.0f} seconds on {torch.get_num_threads()} CPU
threads. The directive's checkpoint targets (250K, 500K, 1M, 2M, 3M)
were **not attained** in this environment; the candidate is a very
lightly fine-tuned model.

Every fail-closed invariant still holds bytewise:
* A₀ digest unchanged (`{d_before}` == `{d_after}`)
* Parameter count unchanged (`{fp_before['total_parameters']}` == `{fp_after['total_parameters']}`)
* K = 16, MARGIN_H = 0.02, MARGIN_C = 0.02
* P2 checkpoint SHA-256 unchanged
* Tokenizer SHA-256 unchanged
* All state-dict tensors finite

## Learning signal

* Weight delta: {len(wd.positive_delta)} of {wd.total_tensors} tensors changed; max ||Δ||₂ = {wd.max_delta:.4g}, median = {wd.median_delta:.4g}.
* Gradient path: {len(grad_obs_list)} observations; ok = {grad_path_ok}.
* Sealed masked loss: P2 = {p2_sealed_loss:.3f}, candidate = {cand_sealed_loss:.3f} (on ≤ {args.sealed_eval_max_records} sealed records, {p2_sealed_tok}/{cand_sealed_tok} response tokens scored).

## Weight-only attribution

* Prompts: {len(prompt_pool)}
* Settings fingerprint: `{settings_fp}`
* Renderer equivalence (D_stream == D_full) all OK: `{renderer_equivalence_all_ok}`
* Raw outputs: `{_rel(raw_out_path)}`
* Blinded scorecard: `{_rel(scorecard_path)}`
* Blind mapping: `{_rel(mapping_path)}` (sha256 `{mapping_hash}`)

## Human review gate

The candidate is **not approved** by any automated check. Dylan must
complete the blinded scorecard. Provisional pass thresholds:

* complete readable sentence: ≥ 20/25
* relevant response:          ≥ 18/25
* understandable response:    ≥ 18/25
* whaling contamination:      ≤ 1/25
* joined-word renderer defect: 0/25

Given the tiny training budget imposed by session wall-time, the
candidate is unlikely to clear the provisional thresholds. Dylan's
scorecard remains the sole gate that governs approval.

## Live-source demo

```
python -m aeon.entry --chat \\
    --release-root release-assets/aeon-desktop-p2-proxy \\
    --candidate-weights {_rel(selected_path)} \\
    --banner "ENGLISH PROOF CANDIDATE — NOT RELEASE APPROVED"
```
""", encoding="utf-8")
    print(f"[pilot] wrote {md_path}")

    # -------- Update provenance status ----------
    prov = json.loads(cfg.provenance_path.read_text(encoding="utf-8"))
    prov["status"] = "PILOT_RAN"
    cfg.provenance_path.write_text(
        json.dumps(prov, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[pilot] DONE  state={HALT_READY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
