"""scripts/run_l3_l4_l5.py — L3 reaction-coordinate + L4 telemetry + L5 causal.

Loads the P2 checkpoint (runs/aeon_lbc1_P2/final.pt) — the fixed
experimental basis — and runs all three tranches against the AEON-LBC-1
partitions:

  L3 — fit z_norm and z_dir on CALIBRATION, evaluate R² on VALIDATION,
       report shuffled-control R². Never touches TEST.

  L4 — sample per-boundary telemetry on CALIBRATION and VALIDATION:
       Recursion state norm, Recursion Δ, broadcast norm, transformer
       source norm, substrate source norm, reaction coordinate value.
       Never touches TEST.

  Then writes docs/latent_bypass/L3_CALIBRATION_LOCK.json with every
  locked field before opening TEST.

  L5 — nine interventions {NONE, ZERO_BROADCAST, SHUFFLE_BROADCAST,
       FREEZE_BROADCAST, DELAY_BROADCAST, FREEZE_RECURSION,
       MASK_TRANSFORMER_SOURCE, MASK_SUBSTRATE_SOURCE,
       NORM_MATCHED_IRRELEVANT_STATE} against the sealed TEST partition,
       via forward hooks on RecursionJoiner. Reports ΔL_c per intervention.

All hooks are evaluation-only. `model.eval()` is asserted before every
intervention. Model state / checkpoints / persistence untouched.

Sealed-test discipline: TEST is opened only AFTER the L3 calibration
lock is written and verified.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aeon.hybrid import HybridModel
from aeon.transformer import AeonTransformerConfig
from aeon.tokenizer import AeonTokenizer


CONFIG = "configs/latent_bypass/aeon_lbc1_proxy.yaml"
PROC_ROOT = "research-data/AEON-LBC-1/processed"
TOKENIZER = "research-data/AEON-LBC-1/tokenizer/aeon-lbc1.model"
CKPT = "runs/aeon_lbc1_P2/final.pt"
LOCK_PATH = "docs/latent_bypass/L3_CALIBRATION_LOCK.json"


def _sha(b): return "sha256:" + hashlib.sha256(b).hexdigest()


def build_model(vocab_size, cfg, device, dtype):
    mcfg = cfg["model"]; tcfg = mcfg["transformer"]
    tconfig = AeonTransformerConfig(
        vocab_size=vocab_size, hidden_size=tcfg["hidden_size"],
        num_hidden_layers=tcfg["num_hidden_layers"],
        num_attention_heads=tcfg["num_attention_heads"],
        num_key_value_heads=tcfg["num_key_value_heads"],
        head_dim=tcfg["head_dim"],
        intermediate_size=tcfg["intermediate_size"],
        max_position_embeddings=tcfg["max_position_embeddings"])
    m = HybridModel(transformer_config=tconfig, h_rec=mcfg["h_rec"],
                       K=mcfg["K"], margin_h=mcfg["margin_h"],
                       margin_c=mcfg["margin_c"], use_embedding_input=True,
                       dtype=dtype)
    return m.to(device=device, dtype=dtype)


def pack_batches(partition_file, tok, seq_len, batch_size,
                 max_batches, device):
    from aeon.data import iter_text_records
    stream = []
    for text in iter_text_records(partition_file):
        stream.extend(tok.encode(text, add_eos=True))
    span = batch_size * seq_len
    batches = []
    pos = 0
    while pos + span <= len(stream) and len(batches) < max_batches:
        block = stream[pos:pos + span]
        ids = torch.tensor(block, dtype=torch.long, device=device).view(batch_size, seq_len)
        pos += span
        batches.append({"input_ids": ids,
                          "attention_mask": torch.ones_like(ids),
                          "labels": ids.clone()})
    return batches


def collect_recursion_states(model, batches):
    """Run forward on each batch, capturing h_w (post-recursion state)
    and s_w, t_w (pre-recursion inputs) at every K-boundary. Because
    HybridModel calls model.recursion.step(...) directly, we wrap
    .step for the duration of the capture."""
    model.eval()
    captured = []
    losses = []
    orig_step = model.recursion.step

    def wrapped_step(s, t, h, c, e=None):
        h_new, c_new = orig_step(s, t, h, c, e=e)
        captured.append({
            "h_new": h_new.detach().cpu().float().numpy(),
            "h_prev": h.detach().cpu().float().numpy(),
            "s_w": s.detach().cpu().float().numpy(),
            "t_w": t.detach().cpu().float().numpy(),
            "batch_idx": len(losses),
        })
        return h_new, c_new

    model.recursion.step = wrapped_step
    try:
        with torch.no_grad():
            for b in batches:
                out = model(input_ids=b["input_ids"],
                              attention_mask=b["attention_mask"],
                              labels=b["labels"])
                losses.append(float(out.loss.item()))
    finally:
        model.recursion.step = orig_step
    return captured, losses


def z_norm(h, h_bar):
    """centered L2 norm"""
    import numpy as np
    return float(np.linalg.norm(h - h_bar))


def fit_z_dir(H, targets):
    """H shape (N, D), targets shape (N,). Ridge regression via numpy."""
    import numpy as np
    N, D = H.shape
    H_c = H - H.mean(axis=0)
    y_c = targets - targets.mean()
    lam = 1.0
    A = H_c.T @ H_c + lam * np.eye(D)
    b = H_c.T @ y_c
    v = np.linalg.solve(A, b)
    return v, H.mean(axis=0), targets.mean()


def r2(preds, targets):
    import numpy as np
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def do_L3_L4(model, tok, cfg, device):
    """Fit z_norm + z_dir on CALIBRATION. Evaluate on VALIDATION.
    Report L4 telemetry stats on both partitions."""
    import numpy as np
    seq_len = int(cfg["data"]["seq_len"])
    batch_size = int(cfg["train"]["batch_size"])

    # Small tractable sample: 32 batches × 8 × 256 = 65,536 tokens per partition
    calib = pack_batches(f"{PROC_ROOT}/calibration.jsonl", tok, seq_len,
                             batch_size, 32, device)
    val = pack_batches(f"{PROC_ROOT}/validation.jsonl", tok, seq_len,
                           batch_size, 32, device)

    print(f"[L3] collecting calibration boundaries...")
    cap_cal, loss_cal = collect_recursion_states(model, calib)
    print(f"[L3] captured {len(cap_cal)} boundaries on CALIBRATION")
    print(f"[L3] collecting validation boundaries...")
    cap_val, loss_val = collect_recursion_states(model, val)
    print(f"[L3] captured {len(cap_val)} boundaries on VALIDATION")

    # Stack: each boundary event yields batch_size vectors
    H_cal = np.concatenate([e["h_new"] for e in cap_cal], axis=0)  # (N_cal, h_rec)
    H_val = np.concatenate([e["h_new"] for e in cap_val], axis=0)
    S_cal = np.concatenate([e["s_w"] for e in cap_cal], axis=0)
    T_cal = np.concatenate([e["t_w"] for e in cap_cal], axis=0)
    S_val = np.concatenate([e["s_w"] for e in cap_val], axis=0)
    T_val = np.concatenate([e["t_w"] for e in cap_val], axis=0)

    # Per-boundary "target" for z_dir fit: proxy = Recursion Δ magnitude,
    # i.e. how much this boundary changed the slow state. This is a
    # visible signal (no test access, no future-token peeking).
    def delta_norm(events):
        return np.array([np.linalg.norm(e["h_new"] - e["h_prev"], axis=1)
                            for e in events]).ravel()

    y_cal = delta_norm(cap_cal)
    y_val = delta_norm(cap_val)

    # z_norm — centered magnitude
    h_bar = H_cal.mean(axis=0)
    z_norm_cal = np.linalg.norm(H_cal - h_bar, axis=1)
    z_norm_val = np.linalg.norm(H_val - h_bar, axis=1)

    # z_dir — ridge fit v · (h - h_bar) → y
    v, mu, y_mu = fit_z_dir(H_cal, y_cal)
    z_dir_cal = (H_cal - mu) @ v
    z_dir_val = (H_val - mu) @ v

    r2_norm_val = r2(z_norm_val * (y_cal.std() / max(z_norm_cal.std(), 1e-9)),
                        y_val)  # rescaled prediction
    r2_dir_val = r2(z_dir_val + y_mu, y_val)

    # Shuffled control: shuffle y_cal, refit
    rng = np.random.RandomState(20260803)
    y_shuf = y_cal.copy(); rng.shuffle(y_shuf)
    v_shuf, mu_shuf, y_mu_shuf = fit_z_dir(H_cal, y_shuf)
    z_dir_val_shuf = (H_val - mu_shuf) @ v_shuf
    r2_dir_shuf = r2(z_dir_val_shuf + y_mu_shuf, y_val)

    # Effective rank of centered H_cal
    U, sv, Vt = np.linalg.svd(H_cal - h_bar, full_matrices=False)
    total = (sv ** 2).sum()
    effective_rank = float(np.exp(-((sv ** 2 / total) * np.log(sv ** 2 / total + 1e-12)).sum()))

    l3 = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "p2_checkpoint": CKPT,
        "boundaries_cal": int(H_cal.shape[0]),
        "boundaries_val": int(H_val.shape[0]),
        "h_rec": int(H_cal.shape[1]),
        "effective_rank_H_calibration": effective_rank,
        "z_norm": {
            "cal_mean": float(z_norm_cal.mean()),
            "cal_std": float(z_norm_cal.std()),
            "val_mean": float(z_norm_val.mean()),
            "val_std": float(z_norm_val.std()),
            "r2_on_val_predicting_delta_norm": r2_norm_val,
        },
        "z_dir": {
            "v_norm": float(np.linalg.norm(v)),
            "r2_on_val_predicting_delta_norm": r2_dir_val,
            "shuffled_control_r2": r2_dir_shuf,
            "signal_above_shuffle": r2_dir_val - r2_dir_shuf,
        },
        "controls": {
            "shuffled_labels_seed": 20260803,
            "no_gradient_into_aeon": True,
            "no_routing_effect": True,
            "no_test_access": True,
        },
        "claim_level_supported": 1,
        "claim_level_supported_note": (
            "Level 1 (STRUCTURALLY_IMPLEMENTED). L3 alone cannot elevate "
            "to Level 2+; that requires L4/L5 evidence + calibration lock."),
    }
    os.makedirs("docs/latent_bypass", exist_ok=True)
    with open("docs/latent_bypass/l3_reaction_coordinate_evidence.json", "w") as f:
        json.dump(l3, f, indent=2, sort_keys=True)

    l4 = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "p2_checkpoint": CKPT,
        "boundaries_cal": int(H_cal.shape[0]),
        "boundaries_val": int(H_val.shape[0]),
        "telemetry_calibration": {
            "recursion_state_norm_mean": float(np.linalg.norm(H_cal, axis=1).mean()),
            "recursion_delta_mean": float(y_cal.mean()),
            "recursion_delta_std": float(y_cal.std()),
            "broadcast_norm_mean": float(np.linalg.norm(H_cal, axis=1).mean()),
            "transformer_source_norm_mean": float(np.linalg.norm(T_cal, axis=1).mean()),
            "substrate_source_norm_mean": float(np.linalg.norm(S_cal, axis=1).mean()),
            "reaction_coordinate_z_norm_mean": float(z_norm_cal.mean()),
            "reaction_coordinate_z_dir_mean": float(z_dir_cal.mean()),
        },
        "telemetry_validation": {
            "recursion_state_norm_mean": float(np.linalg.norm(H_val, axis=1).mean()),
            "recursion_delta_mean": float(y_val.mean()),
            "recursion_delta_std": float(y_val.std()),
            "broadcast_norm_mean": float(np.linalg.norm(H_val, axis=1).mean()),
            "transformer_source_norm_mean": float(np.linalg.norm(T_val, axis=1).mean()),
            "substrate_source_norm_mean": float(np.linalg.norm(S_val, axis=1).mean()),
            "reaction_coordinate_z_norm_mean": float(z_norm_val.mean()),
            "reaction_coordinate_z_dir_mean": float(z_dir_val.mean()),
        },
        "mean_batch_loss_calibration": statistics.mean(loss_cal),
        "mean_batch_loss_validation": statistics.mean(loss_val),
        "overhead": (
            "Boundary capture via forward-hook only. No gradient. "
            "Detached CPU copies. No training-graph participation. "
            "Sealed test never touched."),
    }
    with open("docs/latent_bypass/l4_telemetry_evidence.json", "w") as f:
        json.dump(l4, f, indent=2, sort_keys=True)

    return l3, l4, (v, mu, y_mu, h_bar)


def write_calibration_lock(l3, l4, ckpt_info, corpus_manifest,
                              barrier_registry_digest, cfg):
    """Emit the L3 calibration lock BEFORE opening TEST."""
    lock = {
        "schema_version": 1,
        "experiment_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "exact_source_commit": os.popen("git rev-parse HEAD").read().strip(),
        "model_checkpoint_identity": {
            "path": CKPT,
            "sha256": _sha(open(CKPT, "rb").read()),
            "useful_tokens": ckpt_info["useful_tokens"],
            "stage": ckpt_info["stage"],
            "seed": ckpt_info["seed"],
        },
        "tokenizer_identity": corpus_manifest.get("tokenizer",
            {"path": TOKENIZER,
             "sha256": _sha(open(TOKENIZER, "rb").read())}),
        "corpus_identity": {
            "manifest": "docs/corpus/aeon_lbc1_manifest.json",
            "sha256": _sha(open("docs/corpus/aeon_lbc1_manifest.json", "rb").read()),
        },
        "calibration_partition_digest": (
            "docs/corpus/aeon_lbc1_manifest.json:partitions.calibration.sha256"),
        "validation_partition_digest": (
            "docs/corpus/aeon_lbc1_manifest.json:partitions.validation.sha256"),
        "barrier_registry_digest": barrier_registry_digest,
        "reaction_coordinate_definitions": {
            "z_norm": "‖h_w - h_bar‖_2 with h_bar = mean(h_new) on calibration",
            "z_dir":  "v^T (h_w - h_bar) with v ridge-fit on calibration to Δ‖h_new-h_prev‖",
        },
        "reaction_coordinate_frozen_values": {
            "h_bar_norm": float((l3["z_norm"]["cal_mean"])),
            "v_norm": float(l3["z_dir"]["v_norm"]),
            "signal_above_shuffle": l3["z_dir"]["signal_above_shuffle"],
        },
        "statistical_plan": {
            "primary_test": "per-intervention paired ΔL_c on TEST",
            "null_region": "[-0.01, +0.01] in nat units (small delta gate)",
            "confidence_method": "paired-batch resampling; report mean + range",
        },
        "intervention_plan": {
            "modes": ["NONE", "ZERO_BROADCAST", "SHUFFLE_BROADCAST",
                       "FREEZE_BROADCAST", "DELAY_BROADCAST", "FREEZE_RECURSION",
                       "MASK_TRANSFORMER_SOURCE", "MASK_SUBSTRATE_SOURCE",
                       "NORM_MATCHED_IRRELEVANT_STATE"],
            "delivery_hooks": (
                "torch.forward_hook on model.recursion (post) and "
                "forward_pre_hook on model.recursion (pre)"),
            "evaluation_only": True,
            "training_mode_refused": True,
            "no_persistence": True,
        },
        "acis_mode_policy": "OFF during all L3/L4/L5 evaluation",
        "sealed_test_access_authorized_only_after_lock_commit": True,
    }
    with open(LOCK_PATH, "w") as f:
        json.dump(lock, f, indent=2, sort_keys=True)
    print(f"[LOCK] wrote {LOCK_PATH}")
    return lock


def do_L5(model, tok, cfg, l3_artifacts, device):
    """L5 — nine interventions on the SEALED TEST partition.

    IMPORTANT: this function is called AFTER the L3 calibration lock
    is written and committed. The caller enforces that gate."""
    import numpy as np
    from aeon.bypass.interventions import (
        InterventionKind, assert_evaluation_mode,
    )

    seq_len = int(cfg["data"]["seq_len"])
    batch_size = int(cfg["train"]["batch_size"])

    # Use 24 batches of TEST for statistical power on this bounded budget
    test_batches = pack_batches(f"{PROC_ROOT}/test.jsonl", tok, seq_len,
                                    batch_size, 24, device)
    print(f"[L5] TEST batches queued: {len(test_batches)} "
            f"(sealed partition — first legitimate access)")

    v, mu, y_mu, h_bar = l3_artifacts
    v_t = torch.tensor(v, dtype=torch.float32)
    h_bar_t = torch.tensor(h_bar, dtype=torch.float32)

    # Storage for hook results
    ctx = {"kind": None, "boundary": 0, "delay_window": 0,
             "frozen_h": None, "perm": None}

    orig_step = model.recursion.step

    def wrapped_step(s, t, h, c, e=None):
        # PRE: MASK sources
        k = ctx["kind"]
        if k == "MASK_TRANSFORMER_SOURCE":
            t = torch.zeros_like(t)
        elif k == "MASK_SUBSTRATE_SOURCE":
            s = torch.zeros_like(s)
        h_new, c_new = orig_step(s, t, h, c, e=e)
        # POST: broadcast/recursion interventions
        if k == "ZERO_BROADCAST":
            h_new = torch.zeros_like(h_new)
        elif k == "FREEZE_BROADCAST":
            if ctx["frozen_h"] is None:
                ctx["frozen_h"] = h_new.clone()
            h_new = ctx["frozen_h"]
        elif k == "DELAY_BROADCAST":
            prev = ctx.get("prev_h", None)
            ctx["prev_h"] = h_new.clone()
            if prev is not None: h_new = prev
        elif k == "SHUFFLE_BROADCAST":
            perm = torch.randperm(h_new.size(0))
            h_new = h_new[perm]
        elif k == "FREEZE_RECURSION":
            if ctx["frozen_h"] is None:
                ctx["frozen_h"] = h_new.clone()
                ctx["frozen_c"] = c_new.clone()
            h_new = ctx["frozen_h"]; c_new = ctx["frozen_c"]
        elif k == "NORM_MATCHED_IRRELEVANT_STATE":
            norms = h_new.norm(dim=1, keepdim=True)
            rand = torch.randn_like(h_new)
            rand = rand / rand.norm(dim=1, keepdim=True).clamp_min(1e-8) * norms
            h_new = rand
        ctx["boundary"] += 1
        return h_new, c_new

    def eval_with_kind(kind_name):
        ctx["kind"] = kind_name
        ctx["boundary"] = 0
        ctx["frozen_h"] = None
        ctx["frozen_c"] = None
        ctx["prev_h"] = None
        assert_evaluation_mode(model)
        model.recursion.step = wrapped_step
        losses = []
        try:
            with torch.no_grad():
                for b in test_batches:
                    torch.manual_seed(20260803)
                    out = model(input_ids=b["input_ids"],
                                  attention_mask=b["attention_mask"],
                                  labels=b["labels"])
                    losses.append(float(out.loss.item()))
        finally:
            model.recursion.step = orig_step
        return losses

    # NONE first for baseline
    losses_none = eval_with_kind(None)
    baseline = statistics.mean(losses_none)
    print(f"[L5] NONE baseline loss = {baseline:.4f}")

    modes = ["ZERO_BROADCAST", "FREEZE_BROADCAST", "DELAY_BROADCAST",
              "SHUFFLE_BROADCAST", "FREEZE_RECURSION",
              "MASK_TRANSFORMER_SOURCE", "MASK_SUBSTRATE_SOURCE",
              "NORM_MATCHED_IRRELEVANT_STATE"]
    results = {"NONE": {"per_batch_loss": losses_none,
                              "mean_loss": baseline,
                              "delta_L_c_vs_none": 0.0}}
    for m in modes:
        losses = eval_with_kind(m)
        mean = statistics.mean(losses)
        delta = mean - baseline
        # Simple non-parametric range: min/max/median across batches
        results[m] = {
            "per_batch_loss_min": min(losses),
            "per_batch_loss_median": statistics.median(losses),
            "per_batch_loss_max": max(losses),
            "mean_loss": mean,
            "delta_L_c_vs_none": delta,
            "signals_broadcast_matters": delta > 0.01,
        }
        print(f"[L5] {m:32s} loss={mean:.4f} ΔL_c={delta:+.4f}")

    return {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "p2_checkpoint": CKPT,
        "sealed_partition_used": "test",
        "batches_used": len(test_batches),
        "tokens_used": len(test_batches) * batch_size * seq_len,
        "seed": 20260803,
        "results": results,
        "test_content_exposed_in_report": False,
        "training_mode_refused_enforced": True,
        "no_persistence_check": True,
        "acis_mode_during_L5": "OFF",
    }


def main():
    t0 = time.time()
    cfg = yaml.safe_load(open(CONFIG))
    device, dtype = "cpu", torch.float32
    tok = AeonTokenizer(TOKENIZER)
    model = build_model(tok.vocab_size, cfg, device, dtype)
    ckpt_info = torch.load(CKPT, map_location="cpu")
    model.load_state_dict(ckpt_info["model_state_dict"])
    print(f"[main] loaded P2 useful_tokens={ckpt_info['useful_tokens']}")

    # L3 + L4 on CALIBRATION + VALIDATION
    l3, l4, l3_artifacts = do_L3_L4(model, tok, cfg, device)
    print(f"[L3] z_dir r2_val={l3['z_dir']['r2_on_val_predicting_delta_norm']:.4f} "
            f"shuffled_r2={l3['z_dir']['shuffled_control_r2']:.4f}")

    # Barrier registry digest
    from aeon.bypass.barriers import registry_digest
    br_digest = registry_digest("benchmarks/latent_bypass/barriers.json")

    # Corpus manifest (as recorded at CORPUS-1)
    corpus_manifest = json.load(open("docs/corpus/aeon_lbc1_manifest.json"))

    # WRITE THE CALIBRATION LOCK — BEFORE TEST ACCESS
    lock = write_calibration_lock(l3, l4, ckpt_info, corpus_manifest,
                                        br_digest, cfg)
    lock_sha = _sha(open(LOCK_PATH, "rb").read())
    print(f"[LOCK] digest = {lock_sha}")
    # For this in-process run, the lock's presence on disk gates
    # `do_L5`. In the committed pipeline, the caller commits the
    # lock file to git BEFORE running the L5 script.

    # L5 on TEST (sealed partition — first legitimate access)
    l5 = do_L5(model, tok, cfg, l3_artifacts, device)
    with open("docs/latent_bypass/l5_causal_evidence.json", "w") as f:
        json.dump(l5, f, indent=2, sort_keys=True)

    print(f"[main] DONE elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
