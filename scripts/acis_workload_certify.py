"""scripts/acis_workload_certify.py — ACIS workload certification (§13).

Loads the P2 checkpoint, then for each of ACIS OFF / OBSERVE / BUCKET
runs repeated paired evaluation trials against identical batches and
identical RNG. Measures wall time, tokens/second, peak transient
memory (via tracemalloc), event count, and boundary latency.

Then computes:
    overhead_median  = median(mode_time - OFF_time) / median(OFF_time)
    overhead_p95     = p95(mode_time) / p95(OFF_time) - 1

Requires (§13):
    OBSERVE median overhead < 3%
    OBSERVE hard ceiling ≤ 5%
    BUCKET  logit diff = 0, loss diff = 0, gradient diff = 0

Writes:
    docs/acis/acis_workload_evidence.json
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aeon.hybrid import HybridModel
from aeon.transformer import AeonTransformerConfig
from aeon.tokenizer import AeonTokenizer


CONFIG = "configs/latent_bypass/aeon_lbc1_proxy.yaml"
CORPUS = "research-data/AEON-LBC-1/processed/train.jsonl"
TOKENIZER = "research-data/AEON-LBC-1/tokenizer/aeon-lbc1.model"
CKPT = "runs/aeon_lbc1_P2/final.pt"


def build_model(vocab_size, cfg, device, dtype):
    mcfg = cfg["model"]
    tcfg = mcfg["transformer"]
    tconfig = AeonTransformerConfig(
        vocab_size=vocab_size, hidden_size=tcfg["hidden_size"],
        num_hidden_layers=tcfg["num_hidden_layers"],
        num_attention_heads=tcfg["num_attention_heads"],
        num_key_value_heads=tcfg["num_key_value_heads"],
        head_dim=tcfg["head_dim"],
        intermediate_size=tcfg["intermediate_size"],
        max_position_embeddings=tcfg["max_position_embeddings"],
    )
    m = HybridModel(transformer_config=tconfig, h_rec=mcfg["h_rec"],
                       K=mcfg["K"], margin_h=mcfg["margin_h"],
                       margin_c=mcfg["margin_c"], use_embedding_input=True,
                       dtype=dtype)
    return m.to(device=device, dtype=dtype)


def load_p2(model, path):
    st = torch.load(path, map_location="cpu")
    model.load_state_dict(st["model_state_dict"])
    return st


def make_batches(tok, seq_len, batch_size, n_batches):
    from aeon.data import iter_text_records
    stream = []
    for text in iter_text_records(CORPUS):
        stream.extend(tok.encode(text, add_eos=True))
    span = batch_size * seq_len
    batches = []
    pos = 0
    for _ in range(n_batches):
        if pos + span > len(stream): pos = 0
        block = stream[pos:pos + span]
        pos += span
        ids = torch.tensor(block, dtype=torch.long).view(batch_size, seq_len)
        batches.append({"input_ids": ids,
                          "attention_mask": torch.ones_like(ids),
                          "labels": ids.clone()})
    return batches


def _digest_tensor(t):
    b = t.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _digest_gradients(model):
    h = hashlib.sha256()
    for p in model.parameters():
        if p.grad is None:
            h.update(b"NONE")
        else:
            h.update(p.grad.detach().to(torch.float32).contiguous().cpu().numpy().tobytes())
    return "sha256:" + h.hexdigest()


def eval_trial(model, batches, seed, shuttle):
    """Return dict with wall_ms, tokens, event_count (if shuttle), peak_mem."""
    model.eval()
    torch.manual_seed(seed)
    gc.collect()
    tracemalloc.start()
    boundary_latencies_ms = []
    ev_before = 0
    audit_before = 0
    if shuttle is not None:
        ev_before = len(getattr(shuttle, "published", []) or [])
        al = getattr(shuttle, "audit_log", None)
        if al is not None:
            audit_before = len(al.events())

    t0 = time.time()
    tokens = 0
    losses = []
    logit_digest_stack = hashlib.sha256()
    with torch.no_grad():
        for batch in batches:
            tb = time.time()
            out = model(input_ids=batch["input_ids"],
                          attention_mask=batch["attention_mask"],
                          labels=batch["labels"],
                          shuttle=shuttle)
            boundary_latencies_ms.append((time.time() - tb) * 1000)
            losses.append(float(out.loss.item()))
            tokens += batch["input_ids"].numel()
            logit_digest_stack.update(_digest_tensor(out.logits).encode())
    wall_ms = (time.time() - t0) * 1000
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ev_after = 0
    audit_after = 0
    if shuttle is not None:
        ev_after = len(getattr(shuttle, "published", []) or [])
        al = getattr(shuttle, "audit_log", None)
        if al is not None:
            audit_after = len(al.events())
    return {
        "wall_ms": wall_ms,
        "tokens": tokens,
        "tokens_per_sec": tokens / (wall_ms / 1000),
        "mean_loss": statistics.mean(losses),
        "peak_transient_bytes": peak,
        "publish_count_delta": ev_after - ev_before,
        "audit_event_count_delta": audit_after - audit_before,
        "boundary_latency_ms_median": statistics.median(boundary_latencies_ms),
        "boundary_latency_ms_p95": sorted(boundary_latencies_ms)[
            int(len(boundary_latencies_ms) * 0.95)],
        "stacked_logit_digest": "sha256:" + logit_digest_stack.hexdigest(),
    }


def grad_diff_check(model, batches, seed, shuttle):
    """Assert gradient equivalence under this shuttle mode versus OFF."""
    model.train()
    # OFF
    torch.manual_seed(seed)
    for p in model.parameters():
        p.grad = None
    out_off = model(input_ids=batches[0]["input_ids"],
                       attention_mask=batches[0]["attention_mask"],
                       labels=batches[0]["labels"],
                       shuttle=None)
    out_off.loss.backward()
    d_off = _digest_gradients(model)
    # this shuttle
    torch.manual_seed(seed)
    for p in model.parameters():
        p.grad = None
    out_x = model(input_ids=batches[0]["input_ids"],
                     attention_mask=batches[0]["attention_mask"],
                     labels=batches[0]["labels"],
                     shuttle=shuttle)
    out_x.loss.backward()
    d_x = _digest_gradients(model)
    for p in model.parameters():
        p.grad = None
    return d_off == d_x, d_off, d_x


def main():
    t0 = time.time()
    cfg = yaml.safe_load(open(CONFIG))
    device, dtype = "cpu", torch.float32
    seed = int(cfg["train"]["seed"])
    seq_len = int(cfg["data"]["seq_len"])
    batch_size = int(cfg["train"]["batch_size"])

    tok = AeonTokenizer(TOKENIZER)
    model = build_model(tok.vocab_size, cfg, device, dtype)
    ckpt_info = load_p2(model, CKPT)
    print(f"[cert] loaded P2 checkpoint useful_tokens={ckpt_info['useful_tokens']}")

    # Small stable trial workload — 32 batches × 8 batch × 256 seq = 65,536 tokens.
    batches = make_batches(tok, seq_len, batch_size, n_batches=32)
    trials_per_mode = 5

    from aeon.shuttle.routing import StandardAcisShuttle

    results = {"OFF": [], "OBSERVE": [], "BUCKET": []}
    stacked_digests = {}
    for mode in ("OFF", "OBSERVE", "BUCKET"):
        for i in range(trials_per_mode):
            shuttle = None if mode == "OFF" else StandardAcisShuttle(mode=mode)
            r = eval_trial(model, batches, seed=seed + i, shuttle=shuttle)
            results[mode].append(r)
            print(f"[cert] {mode} trial {i+1}/{trials_per_mode}: "
                    f"wall={r['wall_ms']:.0f}ms peak={r['peak_transient_bytes']/1e6:.2f}MB "
                    f"publishes={r['publish_count_delta']} audit_events={r['audit_event_count_delta']}")
            stacked_digests[mode] = r["stacked_logit_digest"]

    # Overhead computation vs OFF
    def med(m): return statistics.median([r["wall_ms"] for r in results[m]])
    def p95(m):
        xs = sorted([r["wall_ms"] for r in results[m]])
        return xs[int(len(xs) * 0.95)]
    off_med = med("OFF"); obs_med = med("OBSERVE"); buk_med = med("BUCKET")
    off_p95 = p95("OFF"); obs_p95 = p95("OBSERVE"); buk_p95 = p95("BUCKET")

    obs_overhead_median = (obs_med - off_med) / off_med
    obs_overhead_p95 = (obs_p95 - off_p95) / off_p95
    buk_overhead_median = (buk_med - off_med) / off_med
    buk_overhead_p95 = (buk_p95 - off_p95) / off_p95

    # Semantic equivalence: stacked logit digest across all batches
    off_stacked = stacked_digests["OFF"]
    obs_matched = stacked_digests["OBSERVE"] == off_stacked
    buk_matched = stacked_digests["BUCKET"] == off_stacked

    # Gradient equivalence on a single batch
    obs_shuttle = StandardAcisShuttle(mode="OBSERVE")
    buk_shuttle = StandardAcisShuttle(mode="BUCKET")
    obs_grad_ok, gd_off_a, gd_obs = grad_diff_check(model, batches, seed, obs_shuttle)
    buk_grad_ok, gd_off_b, gd_buk = grad_diff_check(model, batches, seed, buk_shuttle)

    # Certification decision (§13)
    obs_certified = (obs_matched
                      and obs_overhead_median < 0.03
                      and obs_overhead_p95 < 0.05)
    buk_semantic_ok = (buk_matched and buk_grad_ok)
    # BUCKET default-eligibility requires (a) semantic ok AND (b) real benefit —
    # the addendum requires "real benefit under the previously certified policy".
    # BUCKET_default_recommended = False unless overhead is negative (faster).
    buk_default_recommended = buk_semantic_ok and (buk_overhead_median <= 0)

    evidence = {
        "schema_version": 1,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "p2_checkpoint": CKPT,
        "batches_per_trial": 32,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "tokens_per_trial": 32 * batch_size * seq_len,
        "trials_per_mode": trials_per_mode,
        "OFF": {
            "wall_ms_median": off_med,
            "wall_ms_p95": off_p95,
            "trials": results["OFF"],
        },
        "OBSERVE": {
            "wall_ms_median": obs_med,
            "wall_ms_p95": obs_p95,
            "overhead_median": obs_overhead_median,
            "overhead_p95": obs_overhead_p95,
            "logit_stacked_matches_OFF": obs_matched,
            "gradient_matches_OFF": obs_grad_ok,
            "median_overhead_under_3pct": obs_overhead_median < 0.03,
            "hard_ceiling_under_5pct": obs_overhead_p95 < 0.05,
            "certified": obs_certified,
            "trials": results["OBSERVE"],
        },
        "BUCKET": {
            "wall_ms_median": buk_med,
            "wall_ms_p95": buk_p95,
            "overhead_median": buk_overhead_median,
            "overhead_p95": buk_overhead_p95,
            "logit_stacked_matches_OFF": buk_matched,
            "gradient_matches_OFF": buk_grad_ok,
            "semantic_equivalent_to_OFF": buk_semantic_ok,
            "default_recommended": buk_default_recommended,
            "trials": results["BUCKET"],
        },
        "conveyor_certified": False,
    }
    os.makedirs("docs/acis", exist_ok=True)
    with open("docs/acis/acis_workload_evidence.json", "w") as f:
        json.dump(evidence, f, indent=2, sort_keys=True)
    print()
    print(f"OFF     wall_ms median={off_med:.1f} p95={off_p95:.1f}")
    print(f"OBSERVE wall_ms median={obs_med:.1f} p95={obs_p95:.1f} "
            f"overhead_median={obs_overhead_median*100:.2f}% "
            f"overhead_p95={obs_overhead_p95*100:.2f}% "
            f"certified={obs_certified}")
    print(f"BUCKET  wall_ms median={buk_med:.1f} p95={buk_p95:.1f} "
            f"overhead_median={buk_overhead_median*100:.2f}% "
            f"overhead_p95={buk_overhead_p95*100:.2f}% "
            f"semantic_ok={buk_semantic_ok} default_recommended={buk_default_recommended}")
    print(f"total_elapsed={time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
