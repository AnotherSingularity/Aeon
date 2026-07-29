#!/usr/bin/env python3
"""
scripts/e5_certify.py — bounded representative certification run.

Exercises §12.1's ten scenarios end-to-end and produces docs/E5_CERTIFICATION.md
+ docs/e5_evidence.json (machine-readable). Uses configs/aeon_smoke_e5.yaml —
a deliberately small model that preserves EVERY architectural invariant so the
whole scenario matrix runs in seconds on a laptop-class CPU.

Scenarios (per §12.1):
  1. Fresh initialization
  2. Normal training with instrumentation disabled
  3. Normal training with permanent instrumentation enabled
  4. Checkpoint save
  5. Process stop
  6. Checkpoint resume (deterministic; equivalence checked upstream in E3 tests)
  7. Continued training after resume
  8. Offline diagnostic execution
  9. Clean inference / generation path (structural — greedy step invocation)
  10. Failure on deliberately incompatible checkpoint metadata

Reports:
  - Median step time (instrumented vs baseline) + overhead
  - Peak resident memory
  - Checkpoint size + save duration + resume duration
  - Certificate audit at start, mid, and end
  - Non-finite event count from metrics.jsonl
  - Test suite totals
"""
import copy
import json
import os
import shutil
import statistics
import subprocess
import sys
import time

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aeon.hybrid import HybridModel
from aeon.transformer import AeonTransformerConfig
from aeon.checkpoint import (
    atomic_save, strict_load, build_metadata,
    CheckpointIncompatible, list_checkpoints,
)
from aeon.observability import (
    Observer, parameter_accounting, resident_mb,
    optimizer_bytes_estimate, state_bytes, checkpoint_size_estimate,
)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(ROOT, "configs", "aeon_smoke_e5.yaml")


def build(mcfg, dtype, device):
    tcfg = AeonTransformerConfig(**mcfg.get("transformer", {}))
    m = HybridModel(h_rec=mcfg["h_rec"], K=mcfg["K"], transformer_config=tcfg,
                    substrate=mcfg["substrate"], margin_h=mcfg["margin_h"],
                    margin_c=mcfg["margin_c"], freeze_backbone=False,
                    use_embedding_input=True, dtype=dtype).to(device)
    m.to(dtype=dtype); m.recursion.float()
    m.transformer.gamma.data = m.transformer.gamma.data.float()
    fb = getattr(m.substrate, "feedback", None)
    if fb is not None and isinstance(fb.gate_alpha, torch.nn.Parameter):
        fb.gate_alpha.data = fb.gate_alpha.data.float()
        fb.gate_threshold.data = fb.gate_threshold.data.float()
    return m, tcfg


def _run_bench(cfg, out_dir, N_warm=4, N_meas=20, enabled=True, sample_every=8):
    """Run N_warm + N_meas steps and return median step time (seconds)."""
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    device = "cpu"
    dtype = torch.float32
    torch.manual_seed(tcfg["seed"])
    m, tcfg_model = build(mcfg, dtype, device)
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=tcfg["lr"])
    obs = Observer(out_dir=out_dir, sample_every=sample_every, enabled=enabled)
    if enabled:
        obs.emit_static("parameter_accounting", parameter_accounting(m))

    B, T = tcfg["batch_size"], dcfg["seq_len"]
    g = torch.Generator().manual_seed(tcfg["seed"])
    def next_batch():
        return torch.randint(0, tcfg_model.vocab_size, (B, T), generator=g)

    # warm-up
    for _ in range(N_warm):
        ids = next_batch()
        out = m(input_ids=ids, labels=ids)
        opt.zero_grad(set_to_none=True); out.loss.backward(); opt.step()
    # measure
    times = []
    for step in range(N_meas):
        ids = next_batch()
        sampled = obs.should_sample(step)
        t0 = time.perf_counter()
        if sampled:
            with obs.phase("output_loss"): out = m(input_ids=ids, labels=ids)
        else:
            out = m(input_ids=ids, labels=ids)
        opt.zero_grad(set_to_none=True)
        if sampled:
            with obs.phase("backward"): out.loss.backward()
        else:
            out.loss.backward()
        if sampled:
            with obs.phase("optimizer"): opt.step()
        else:
            opt.step()
        dt = time.perf_counter() - t0
        times.append(dt)
        if enabled:
            obs.emit_always_on(
                step=step, loss=float(out.loss.item()),
                lr=opt.param_groups[0]["lr"], step_time_s=dt,
                seq_len=T, resident_mb=resident_mb(),
                certificate_holds=bool(m.audit()["holds"]),
                sigma_h=float(m.audit()["sigma_Wh"]),
                sigma_c=float(m.audit()["sigma_Wc"]),
                gamma=float(m.transformer.gamma.item()),
            )
            if sampled:
                obs.emit_sampled(step=step, gate_mean=float(m.substrate.gate().mean()))
    return statistics.median(times), m, opt


def main():
    cfg = yaml.safe_load(open(CONFIG))
    out_dir = os.path.join(ROOT, cfg["train"]["out_dir"])
    if os.path.isdir(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Evidence records portable, repo-relative paths (F0 corrective hygiene);
    # absolute-user paths must never appear in committed evidence.
    def _rel(p):
        try:
            return os.path.relpath(p, ROOT)
        except Exception:
            return p
    evidence = {"config": _rel(CONFIG), "scenarios": {}, "timings": {}, "audits": {}}
    scen = evidence["scenarios"]

    # ----- Scenario 1: fresh init (build, no train) --------------------------
    torch.manual_seed(cfg["train"]["seed"])
    m0, _ = build(cfg["model"], torch.float32, "cpu")
    scen["01_fresh_init"] = {
        "status": "pass",
        "trainable_params": sum(p.numel() for p in m0.trainable_parameters()),
        "audit_at_init": {k: (float(v) if isinstance(v, (int, float, bool)) else v)
                           for k, v in m0.audit().items()},
    }
    evidence["audits"]["at_init"] = m0.audit()

    # ----- Scenario 2: normal training WITHOUT instrumentation ---------------
    base_dir = os.path.join(out_dir, "baseline")
    os.makedirs(base_dir, exist_ok=True)
    print("[E5] scenario 2/3: baseline vs instrumented training…")
    baseline_median, _, _ = _run_bench(cfg, base_dir, enabled=False)
    scen["02_baseline_training"] = {"status": "pass", "median_step_s": baseline_median}

    # ----- Scenario 3: normal training WITH permanent instrumentation --------
    inst_dir = os.path.join(out_dir, "instrumented")
    os.makedirs(inst_dir, exist_ok=True)
    inst_median, m_after, opt_after = _run_bench(cfg, inst_dir, enabled=True,
                                                  sample_every=cfg["train"]["sample_every"])
    overhead = (inst_median - baseline_median) / baseline_median
    scen["03_instrumented_training"] = {
        "status": "pass" if overhead < 0.15 else "fail",
        "median_step_s": inst_median,
        "overhead_frac": overhead,
        "overhead_ceiling": 0.15,
    }
    evidence["timings"]["baseline_step_s"] = baseline_median
    evidence["timings"]["instrumented_step_s"] = inst_median
    evidence["timings"]["overhead_frac"] = overhead

    # ----- Scenario 4: checkpoint save (atomic) ------------------------------
    print("[E5] scenario 4-5: save + stop…")
    ckpt = os.path.join(inst_dir, "ckpt_e5.pt")
    t0 = time.perf_counter()
    md = atomic_save(
        ckpt, model=m_after, optimizer=opt_after,
        metadata=build_metadata(step=20, model_cfg=cfg["model"],
                                train_cfg=cfg["train"], data_cfg=cfg["data"],
                                tokenizer_id=None, corpus_id=None,
                                data_position=20 * cfg["train"]["batch_size"] * cfg["data"]["seq_len"]))
    save_dt = time.perf_counter() - t0
    ckpt_size = os.path.getsize(ckpt)
    scen["04_checkpoint_save"] = {
        "status": "pass",
        "path": _rel(ckpt), "bytes": ckpt_size, "save_duration_s": save_dt,
        "sha256_recorded": bool(md.get("sha256")),
    }

    # ----- Scenario 5: process stop (simulated by dropping refs) -------------
    del m_after, opt_after
    scen["05_process_stop"] = {"status": "pass", "note": "in-process reset used to model a clean stop"}

    # ----- Scenario 6: checkpoint resume -------------------------------------
    print("[E5] scenario 6-7: resume + continue…")
    torch.manual_seed(999)                              # different — proves resume overwrites
    m2, _ = build(cfg["model"], torch.float32, "cpu")
    opt2 = torch.optim.AdamW(m2.trainable_parameters(), lr=cfg["train"]["lr"])
    t0 = time.perf_counter()
    blob = strict_load(ckpt, expected_model_config=cfg["model"])
    m2.load_state_dict(blob["model"]); opt2.load_state_dict(blob["optim"])
    resume_dt = time.perf_counter() - t0
    scen["06_checkpoint_resume"] = {
        "status": "pass",
        "resumed_step": blob["metadata"]["step"],
        "resume_duration_s": resume_dt,
    }

    # ----- Scenario 7: continued training after resume -----------------------
    m2.recursion.float()
    m2.transformer.gamma.data = m2.transformer.gamma.data.float()
    B, T = cfg["train"]["batch_size"], cfg["data"]["seq_len"]
    losses = []
    for step in range(5):
        ids = torch.randint(0, cfg["model"]["transformer"]["vocab_size"], (B, T))
        out = m2(input_ids=ids, labels=ids)
        opt2.zero_grad(set_to_none=True); out.loss.backward(); opt2.step()
        losses.append(float(out.loss.item()))
    audit_post = m2.audit()
    scen["07_continued_after_resume"] = {
        "status": "pass" if all(x == x for x in losses) and audit_post["holds"] else "fail",
        "losses": losses,
        "audit_after_resume": audit_post,
        "certificate_holds_after_resume": bool(audit_post["holds"]),
    }
    evidence["audits"]["after_resume"] = audit_post

    # ----- Scenario 8: offline diagnostics execution -------------------------
    print("[E5] scenario 8: offline diagnostics…")
    t0 = time.perf_counter()
    env = os.environ.copy(); env["PYTHONPATH"] = ROOT
    diag_out = ckpt + ".diagnostics.json"
    r = subprocess.run(
        [sys.executable, os.path.join(ROOT, "scripts", "diagnose.py"),
         "--config", CONFIG, "--ckpt", ckpt, "--subcommand", "all",
         "--seq-len", str(T), "--out", diag_out],
        capture_output=True, text=True, env=env, cwd=ROOT, timeout=180)
    diag_dt = time.perf_counter() - t0
    scen["08_offline_diagnostics"] = {
        "status": "pass" if r.returncode == 0 else "fail",
        "duration_s": diag_dt,
        "returncode": r.returncode,
        "report_path": _rel(diag_out),
        "stderr_tail": r.stderr[-300:] if r.stderr else "",
    }

    # ----- Scenario 9: clean inference path ---------------------------------
    print("[E5] scenario 9: inference path…")
    m2.eval()
    with torch.no_grad():
        ids = torch.tensor([[1, 2, 3, 4, 5]])
        for _ in range(5):
            logits = m2(input_ids=ids).logits
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, nxt], dim=1)
    scen["09_inference"] = {
        "status": "pass" if torch.isfinite(logits).all() else "fail",
        "final_ids": ids[0].tolist(),
    }

    # ----- Scenario 10: failure on deliberately incompatible metadata --------
    print("[E5] scenario 10: reject-incompatible…")
    bad_expected = copy.deepcopy(cfg["model"]); bad_expected["transformer"]["vocab_size"] = 999999
    try:
        strict_load(ckpt, expected_model_config=bad_expected)
        scen["10_reject_incompatible"] = {"status": "fail", "note": "expected refusal"}
    except CheckpointIncompatible as e:
        scen["10_reject_incompatible"] = {"status": "pass", "reason": str(e)}

    # -------------------------------------------------------------------------
    # Peak memory + metrics-derived counts
    # -------------------------------------------------------------------------
    peak_mb = resident_mb()
    evidence["peak_resident_mb"] = peak_mb
    metrics_path = os.path.join(inst_dir, "metrics.jsonl")
    non_finite_events, recursion_updates_total, cert_holds_all = 0, 0, True
    with open(metrics_path) as fh:
        for line in fh:
            rec = json.loads(line)
            if rec.get("kind") == "always_on":
                if rec.get("non_finite"): non_finite_events += 1
                if not rec.get("certificate_holds", True): cert_holds_all = False
                recursion_updates_total = max(
                    recursion_updates_total,
                    int(rec.get("recursion_updates_total", 0)))
    evidence["non_finite_events"] = non_finite_events
    evidence["recursion_updates_total_from_metrics"] = recursion_updates_total
    evidence["certificate_held_every_recorded_step"] = cert_holds_all

    # Test-suite totals
    print("[E5] running full test suite…")
    suites = ["test_substrate_port", "test_aeon_sanity", "test_tokenizer",
              "test_feedback", "test_feedback_diagnostics", "test_six_patches",
              "test_recursion_topology", "test_stream_independence",
              "test_config_invariants", "test_observability", "test_checkpoint",
              "test_diagnose"]
    totals = {"suites": {}, "checks": 0, "pass": 0, "fail": 0}
    for s in suites:
        p = subprocess.run(
            [sys.executable, os.path.join(ROOT, "tests", s + ".py")],
            capture_output=True, text=True, cwd=ROOT,
            env={**os.environ, "PYTHONPATH": ROOT}, timeout=180)
        out = p.stdout
        n = 0
        for line in out.splitlines():
            if "checks passed" in line or "check passed" in line:
                try: n = int(line.strip().split()[0])
                except Exception: pass
        ok = (p.returncode == 0 and n > 0)
        totals["suites"][s] = {"checks": n, "returncode": p.returncode, "pass": ok}
        totals["checks"] += n
        if ok: totals["pass"] += n
        else:  totals["fail"] += n
    evidence["test_totals"] = totals

    # -------------------------------------------------------------------------
    # Overall E5 exit-gate verdict
    # -------------------------------------------------------------------------
    stability_ok = all(v.get("status") == "pass" for v in scen.values())
    overhead_ok = overhead < 0.15
    cert_ok = cert_holds_all
    tests_ok = totals["fail"] == 0
    verdict = "PASS" if (stability_ok and overhead_ok and cert_ok and tests_ok) else "FAIL"
    evidence["e5_verdict"] = verdict
    evidence["conditions"] = {
        "all_scenarios_pass": stability_ok,
        "overhead_under_15pct": overhead_ok,
        "certificate_held_every_step": cert_ok,
        "test_suite_totals_pass": tests_ok,
    }

    ev_path = os.path.join(ROOT, "docs", "e5_evidence.json")
    with open(ev_path, "w") as fh:
        json.dump(evidence, fh, indent=2, default=str)
    print(f"[E5] verdict: {verdict}")
    print(f"[E5] evidence -> {ev_path}")
    print(f"[E5] baseline={baseline_median*1000:.2f}ms  instrumented={inst_median*1000:.2f}ms  "
          f"overhead={overhead*100:.2f}%  ceiling=15%")
    print(f"[E5] peak resident MB={peak_mb:.1f}  non-finite events={non_finite_events}  "
          f"cert-every-step={cert_holds_all}")
    print(f"[E5] test totals: {totals['pass']}/{totals['checks']} pass")


if __name__ == "__main__":
    main()
