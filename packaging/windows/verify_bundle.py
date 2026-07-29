"""verify_bundle.py — post-build smoke tests on the frozen bundle (§W5).

Run on Windows after PyInstaller has produced dist/Aeon:

    python packaging\\windows\\verify_bundle.py --bundle dist\\Aeon

The tests exercise Aeon.exe in every dispatch mode without ever needing a
console window for the user. Any failure exits non-zero and blocks the
installer build.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FAIL: list = []


def _run(argv, *, timeout=60, expect=0):
    """Run a subprocess with no shell; return (rc, stdout, stderr)."""
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                        stdin=subprocess.DEVNULL, shell=False)
    if p.returncode != expect:
        FAIL.append({"argv": argv, "rc": p.returncode, "expected": expect,
                      "stdout_tail": p.stdout[-400:], "stderr_tail": p.stderr[-400:]})
    return p.returncode, p.stdout, p.stderr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="path to dist/Aeon (onedir root)")
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()
    exe = bundle / ("Aeon.exe" if os.name == "nt" else "Aeon")
    if not exe.exists():
        print(f"[verify] bundle exe missing: {exe}", file=sys.stderr)
        sys.exit(1)

    # 1. --version reports metadata and exits 0
    rc, out, err = _run([str(exe), "--version"])
    if rc == 0:
        try:
            meta = json.loads(out)
            for k in ("semantic_version", "source_commit", "build_type"):
                assert k in meta, f"missing {k} in --version"
        except Exception as e:
            FAIL.append({"stage": "--version parse", "detail": str(e)})

    # 2. --verify-installation runs; either passes or reports a structured error
    rc, out, err = _run([str(exe), "--verify-installation"], expect=None)  # accept any RC
    try:
        json.loads(out)
    except Exception as e:
        FAIL.append({"stage": "--verify-installation json", "detail": str(e)})

    # 3. --validate-config on a bundled config (relative to installed_resource_root)
    for cfg in ("configs/aeon_smoke_e5.yaml",):
        cfg_path = bundle / cfg
        if cfg_path.exists():
            rc, _, _ = _run([str(exe), "--validate-config", str(cfg_path)], expect=None)

    # 4. Tiny worker forward pass — construct a minimal job and drive one step.
    #    Uses AEON_DATA_DIR temp to keep the smoke isolated.
    with tempfile.TemporaryDirectory() as d:
        env = os.environ.copy()
        env["AEON_DATA_DIR"] = d
        # Build a tiny synthetic config on disk (matches configs/aeon_smoke_e5)
        tiny_cfg = Path(d) / "tiny.yaml"
        tiny_cfg.write_text("""
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
  max_steps: 2
  batch_size: 1
  grad_clip: 1.0
  log_every: 1
  ckpt_every: 2
  out_dir: runs/verify_bundle
  resume: false
  seed: 0
  sample_every: 100
  observability: true
""", encoding="utf-8")
        job_dir = Path(d) / "job1"; job_dir.mkdir()
        job = {
            "job_id": "smoke", "job_dir": str(job_dir), "config_path": str(tiny_cfg),
            "tokenizer_path": None, "corpus_path": None,
            "checkpoint_dir": str(Path(d) / "ck"),
            "metrics_dir": str(Path(d) / "m"),
            "audit_dir": str(Path(d) / "a"),
            "runtime_policy_id": "p", "security_policy_id": "s",
            "checkpoint_policy": {}, "created_at": time.time(),
            "aeon_source_commit": "smoke", "aeon_release": "0.2.3",
        }
        job_file = job_dir / "job.json"
        job_file.write_text(json.dumps(job), encoding="utf-8")
        rc, out, err = subprocess_run_env([str(exe), "--worker", str(job_file)],
                                            env=env, timeout=120)
        if rc != 0:
            FAIL.append({"stage": "worker", "rc": rc, "stderr_tail": err[-500:]})

    if FAIL:
        print(json.dumps({"ok": False, "failures": FAIL}, indent=2))
        sys.exit(2)
    print(json.dumps({"ok": True, "checks": ["version", "verify-installation",
                                              "validate-config", "worker"]}))


def subprocess_run_env(argv, env, timeout):
    p = subprocess.run(argv, env=env, capture_output=True, text=True,
                        timeout=timeout, stdin=subprocess.DEVNULL, shell=False)
    return p.returncode, p.stdout, p.stderr


if __name__ == "__main__":
    main()
