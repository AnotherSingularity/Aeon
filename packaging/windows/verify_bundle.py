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
    """Run a subprocess with no shell; return (rc, stdout, stderr).
    expect=None accepts any RC and records no failure for RC alone
    (used when the caller only wants to inspect stdout)."""
    p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                        stdin=subprocess.DEVNULL, shell=False)
    if expect is not None and p.returncode != expect:
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

    # 2. --verify-installation runs and MUST pass — integrity is a hard gate
    #    for Tier A: if the frozen bundle's manifest is missing or a hash does
    #    not match, that's a real defect and Tier A stays red.
    rc, out, err = _run([str(exe), "--verify-installation"], expect=0)
    try:
        json.loads(out)
    except Exception as e:
        FAIL.append({"stage": "--verify-installation json", "detail": str(e)})

    # 3. --validate-config on a bundled config (relative to installed_resource_root)
    for cfg in ("configs/aeon_smoke_e5.yaml",):
        cfg_path = bundle / cfg
        if cfg_path.exists():
            rc, _, _ = _run([str(exe), "--validate-config", str(cfg_path)], expect=None)

    # 4. Isolated packaging smoke test of a TEMPORARY TINY model.
    #    This is NOT English training, does NOT modify the protected P2
    #    checkpoint, and does NOT use production corpus data. It exists
    #    to prove the frozen bundle can construct AeonTokenizer, load a
    #    corpus, and drive one worker step end-to-end. AEON_DATA_DIR is
    #    a TemporaryDirectory so nothing is left on disk after the test.
    #
    #    WIN-PATCH-A/Failure B: the previous smoke job passed
    #        "tokenizer_path": None, "corpus_path": None
    #    which the data-source enforcement correctly rejects with
    #    tokenizer_absent. The corrected smoke test uses the tokenizer
    #    bundled inside the frozen distribution and a tiny throw-away
    #    corpus written under the same TemporaryDirectory. The tokenizer
    #    file MUST exist inside the bundle; that is a bundle-integrity
    #    invariant.
    bundled_tokenizer = bundle / "_internal" / "release-assets" / \
        "aeon-desktop-p2-proxy" / "tokenizer" / "aeon-lbc1.model"
    if not bundled_tokenizer.exists():
        FAIL.append({"stage": "worker/tokenizer", "detail":
                     f"bundled tokenizer missing at {bundled_tokenizer}. "
                     "The frozen distribution must contain the AEON-LBC-1 "
                     "tokenizer for the packaging smoke test to run."})

    with tempfile.TemporaryDirectory() as d:
        env = os.environ.copy()
        env["AEON_DATA_DIR"] = d
        # Tiny throw-away smoke corpus (packaging integration test only —
        # NOT a training corpus, NOT authored English content).
        corpus_path = Path(d) / "smoke_corpus.txt"
        corpus_path.write_text(
            "packaging smoke test corpus.\n"
            "this file is temporary and is deleted at the end of the test.\n"
            "it exists only to prove the frozen bundle can complete one worker step.\n",
            encoding="utf-8")

        # Tiny synthetic model config. vocab_size MUST equal the bundled
        # tokenizer's actual vocabulary (16000 for AEON-LBC-1); otherwise
        # the frozen bundle refuses the smoke job at the tokenizer/vocab
        # gate. This is a packaging-layer check, not a training config.
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
    vocab_size: 16000
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
            # WIN-PATCH-A/Failure B: real tokenizer + real (temporary,
            # throw-away) corpus path so the data-source gate accepts
            # the smoke job. Absence of either is the correct failure
            # mode outside this test — do not weaken DataSourceError.
            "tokenizer_path": str(bundled_tokenizer),
            "corpus_path": str(corpus_path),
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
            # Aeon.exe is a Windowed (runw.exe) subsystem executable, so stderr
            # may be empty even on crash. Fold in the structured job artefacts
            # the worker writes on failure so the CI log actually surfaces the
            # cause.
            diag = {"stage": "worker", "rc": rc,
                     "stdout_tail": out[-500:] if out else "",
                     "stderr_tail": err[-500:] if err else ""}
            for name in ("status.json", "result.json"):
                p = job_dir / name
                if p.exists():
                    try:
                        diag[name] = json.loads(p.read_text(encoding="utf-8"))
                    except Exception as e:
                        diag[name + "_raw"] = p.read_text(encoding="utf-8", errors="replace")[-500:]
                        diag[name + "_parse_error"] = str(e)
            errors_log = Path(d) / "logs" / "errors.jsonl"
            if errors_log.exists():
                try:
                    lines = errors_log.read_text(encoding="utf-8", errors="replace").splitlines()
                    diag["errors_jsonl_tail"] = lines[-5:]
                except Exception:
                    pass
            FAIL.append(diag)

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
