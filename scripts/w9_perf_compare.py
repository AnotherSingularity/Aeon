"""w9_perf_compare.py — paired frozen-vs-source performance trials for Tier B.

Runs alternating trials of a bounded, deterministic training fixture on:

  * the frozen Aeon.exe launched with --worker
  * the source-mode `python -m aeon.job.worker` on the same host

Records every trial. Reports median (not min) and packaging overhead ratio.
Refuses to declare "no regression" without at least N=6 alternating trials.

The output JSON is consumed by .github/workflows/windows-certification.yml
step B3.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def _run_trial(exe_argv: list[str], workdir: Path, *, kind: str) -> dict:
    """Run one bounded worker trial. Returns {kind, ok, rc, elapsed_s, ...}."""
    t0 = time.monotonic()
    p = subprocess.run(exe_argv, capture_output=True, text=True,
                        cwd=str(workdir), timeout=300,
                        stdin=subprocess.DEVNULL, shell=False)
    dt = time.monotonic() - t0
    # Worker prints per-step metrics as JSON lines; parse the last one.
    step_time = None
    tokens_per_sec = None
    peak_mem_mb = None
    for line in reversed(p.stdout.splitlines()):
        line = line.strip()
        if not line or not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
        except Exception:
            continue
        step_time = row.get("step_time_ms")
        tokens_per_sec = row.get("tokens_per_sec")
        peak_mem_mb = row.get("peak_mem_mb")
        break
    return {
        "kind": kind,
        "ok": p.returncode == 0,
        "rc": p.returncode,
        "elapsed_s": dt,
        "step_time_ms": step_time,
        "tokens_per_sec": tokens_per_sec,
        "peak_mem_mb": peak_mem_mb,
        "stderr_tail": p.stderr[-400:] if p.stderr else "",
    }


def _median(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-exe", required=True,
                     help="Absolute path to the installed Aeon.exe")
    ap.add_argument("--trials", type=int, default=6,
                     help="Number of alternating source/frozen trials (>=6)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.trials < 6:
        print(f"[w9_perf] --trials must be >= 6 (got {args.trials})", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as td:
        workdir = Path(td)
        os.environ["AEON_DATA_DIR"] = str(workdir)

        frozen_exe = Path(args.frozen_exe).resolve()
        if not frozen_exe.exists():
            print(f"[w9_perf] frozen exe missing: {frozen_exe}", file=sys.stderr)
            return 3

        # Build a bounded fixture worker (uses the same schema
        # verify_bundle.py uses so we exercise a real code path).
        cfg = workdir / "tiny.yaml"
        cfg.write_text(_TINY_CFG, encoding="utf-8")
        job_dir = workdir / "job"; job_dir.mkdir()
        job = {
            "job_id": "perf", "job_dir": str(job_dir),
            "config_path": str(cfg),
            "tokenizer_path": None, "corpus_path": None,
            "checkpoint_dir": str(workdir / "ck"),
            "metrics_dir": str(workdir / "m"),
            "audit_dir": str(workdir / "a"),
            "runtime_policy_id": "p", "security_policy_id": "s",
            "checkpoint_policy": {}, "created_at": time.time(),
            "aeon_source_commit": "perf", "aeon_release": "0.2.3",
        }
        (job_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
        argv_frozen = [str(frozen_exe), "--worker", str(job_dir / "job.json")]
        argv_source = [sys.executable, "-m", "aeon.job.worker",
                        str(job_dir / "job.json")]

        trials = []
        for i in range(args.trials):
            if i % 2 == 0:
                trials.append(_run_trial(argv_source, workdir, kind="source"))
                trials.append(_run_trial(argv_frozen, workdir, kind="frozen"))
            else:
                trials.append(_run_trial(argv_frozen, workdir, kind="frozen"))
                trials.append(_run_trial(argv_source, workdir, kind="source"))

    frozen = [t for t in trials if t["kind"] == "frozen"]
    source = [t for t in trials if t["kind"] == "source"]

    med_step_source = _median([t["step_time_ms"] for t in source])
    med_step_frozen = _median([t["step_time_ms"] for t in frozen])
    overhead_ratio = None
    if med_step_source and med_step_frozen:
        overhead_ratio = med_step_frozen / med_step_source

    result = {
        "trials": trials,
        "medians": {
            "source_step_time_ms": med_step_source,
            "frozen_step_time_ms": med_step_frozen,
            "source_tokens_per_sec": _median([t["tokens_per_sec"] for t in source]),
            "frozen_tokens_per_sec": _median([t["tokens_per_sec"] for t in frozen]),
            "source_peak_mem_mb":   _median([t["peak_mem_mb"] for t in source]),
            "frozen_peak_mem_mb":   _median([t["peak_mem_mb"] for t in frozen]),
        },
        "packaging_overhead_ratio": overhead_ratio,
        "trial_count": len(trials),
        "note": "Median is authoritative. Min-only timing is not reported.",
    }
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)
    print(json.dumps(result["medians"], indent=2))
    return 0


_TINY_CFG = """
model:
  h_rec: 16
  K: 16
  margin_h: 0.98
  margin_c: 0.95
  freeze_backbone: false
  use_embedding_input: true
  dtype: float32
  transformer:
    vocab_size: 64
    hidden_size: 32
    intermediate_size: 64
    num_hidden_layers: 2
    num_attention_heads: 2
    num_key_value_heads: 1
    head_dim: 16
    max_position_embeddings: 64
    rms_norm_eps: 1.0e-5
    rope_theta: 10000.0
    tie_word_embeddings: true
    attention_bias: false
  substrate:
    kind: matrix
    d_in: 16
    d_state: 16
    n_head: 2
    head_size: 8
data:
  seq_len: 32
train:
  lr: 3.0e-4
  weight_decay: 0.0
  max_steps: 4
  batch_size: 1
  grad_clip: 1.0
  log_every: 1
  ckpt_every: 4
  out_dir: runs/perf
  resume: false
  seed: 0
  sample_every: 100
  observability: true
"""


if __name__ == "__main__":
    sys.exit(main())
