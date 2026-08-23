"""scripts/colab/build_notebook.py — assemble Aeon_English_Fluency_Colab.ipynb.

Emits a single valid nbformat-v4 notebook next to itself. Cells are
numbered per the directive and require no shell expertise beyond
'Run cell N'.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def md(*parts) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": [p if p.endswith("\n") else p + "\n" for p in parts]}


def code(*parts) -> dict:
    return {"cell_type": "code", "metadata": {}, "outputs": [],
            "execution_count": None,
            "source": [p if p.endswith("\n") else p + "\n" for p in parts]}


CELLS = [
    md(
        "# Aeon English Fluency — Zero-Cost Colab Campaign",
        "",
        "**Halt state (source repo):** `FREE_COLAB_FLUENCY_BUNDLE_READY`.",
        "",
        "This notebook trains Aeon toward English fluency on the free Colab GPU tier. It:",
        "",
        "1. Mounts your Google Drive",
        "2. Copies the source bundle from Drive into the temporary Colab filesystem",
        "3. Installs pinned dependencies",
        "4. Verifies every bundled file by SHA-256",
        "5. Downloads WikiText-103 raw from the canonical S3 URL, verifies byte size + SHA-256, extracts",
        "6. Detects CUDA (halts if unavailable)",
        "7. Benchmarks Aeon for a short fixed token count and prints projected tokens/hour + estimated sessions",
        "8. Trains **Stage 1** from the protected P2 checkpoint toward 100 million general-English tokens",
        "9. Evaluates WikiText validation at fixed intervals",
        "10. Trains **Stage 2** (Dolly-15k response-masked instruction tuning) starting from the best Stage-1 checkpoint",
        "11. Evaluates on the locked **fresh_eval** subset (contamination-free)",
        "12. Produces raw unedited generations after each major checkpoint",
        "",
        "**Human review gate.** Dylan reviews the raw generations before any release approval. No automatic 'fluent' claim is emitted.",
        "",
        "**Boundaries.** Never modifies parameters during inference. Never calls another model, teacher, corrector, retrieval system, LoRA, adapter, or fallback. Preserves architecture, clocks, K=16, margins, tokenizer, parameter count, state dimensions, state-dict topology.",
        "",
        "**Dry-run.** Set `DRY_RUN = True` in cell 7 to verify the pipeline end-to-end (short benchmark + 5 training steps + one checkpoint + eval sample + generations) without a long training run.",
    ),

    md("## 1 · Mount Google Drive",
       "This cell mounts Drive at `/content/drive` and fails loudly if the "
       "mount does not succeed. Every subsequent path is derived from the "
       "REAL mount point returned here — no `/content/drive/...` path is "
       "created on disk before this cell succeeds."),
    code(
        "import os\n"
        "from google.colab import drive\n"
        "drive.mount('/content/drive')\n"
        "\n"
        "# Fail loudly if the mount did not produce a real MyDrive tree.\n"
        "DRIVE_ROOT = '/content/drive'\n"
        "assert os.path.isdir(DRIVE_ROOT), (\n"
        "    f'Drive mount did not create {DRIVE_ROOT!r}. Re-run this cell.')\n"
        "MYDRIVE = os.path.join(DRIVE_ROOT, 'MyDrive')\n"
        "assert os.path.isdir(MYDRIVE), (\n"
        "    f'Drive mount succeeded but MyDrive is missing at {MYDRIVE!r}. '\n"
        "    'This usually means you cancelled the auth prompt or picked the '\n"
        "    'wrong Google account. Re-run this cell.')\n"
        "print('Drive mount OK ->', MYDRIVE)\n"
    ),

    md("## 2 · Copy the source bundle from Drive & install dependencies",
       "Place `Aeon_English_Fluency_Colab_Bundle.zip` at the root of your Google Drive (MyDrive) first. "
       "This cell refuses to write anything to Drive until it has confirmed the mount is real."),
    code(
        "import os, shutil, subprocess, sys\n"
        "\n"
        "# Guard: cell 1 must have run and succeeded first.\n"
        "assert 'MYDRIVE' in globals(), (\n"
        "    'MYDRIVE is not defined. Run cell 1 (Mount Google Drive) first.')\n"
        "assert os.path.isdir(MYDRIVE), (\n"
        "    f'Drive mount lost between cells; {MYDRIVE!r} no longer exists. '\n"
        "    'Re-run cell 1.')\n"
        "\n"
        "BUNDLE_ZIP = os.path.join(MYDRIVE, 'Aeon_English_Fluency_Colab_Bundle.zip')\n"
        "WORK = '/content/aeon_bundle'\n"
        "assert os.path.exists(BUNDLE_ZIP), (\n"
        "    f'Bundle zip not found at {BUNDLE_ZIP}. Upload '\n"
        "    'Aeon_English_Fluency_Colab_Bundle.zip to the root of your '\n"
        "    'Google Drive (MyDrive) first.')\n"
        "if os.path.exists(WORK): shutil.rmtree(WORK)\n"
        "os.makedirs(WORK, exist_ok=True)\n"
        "subprocess.check_call(['unzip', '-q', BUNDLE_ZIP, '-d', WORK])\n"
        "print('bundle extracted at', WORK)\n"
        "\n"
        "# Install pinned deps (torch already provided by Colab GPU runtime)\n"
        "subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet',\n"
        "    'sentencepiece==0.2.0', 'safetensors', 'pyyaml', 'numpy<2'])\n"
        "print('deps installed')\n"
    ),

    md("## 3 · Verify every bundled file by SHA-256"),
    code(
        "%cd /content/aeon_bundle\n"
        "!python scripts/colab/verify_bundle.py --root .\n"
    ),

    md("## 4 · Download WikiText-103 raw (verified before extraction)",
       "Canonical URL: `https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip`  \n",
       "Expected byte size: `191,984,949`  \n",
       "Expected SHA-256: `91c00ae287f0d699e18605c84afc9e45c192bc6b7797ff8837e5474655a33794`"),
    code(
        "%cd /content/aeon_bundle\n"
        "!python scripts/colab/download_wikitext103.py --out-dir /content/wikitext-103-raw\n"
    ),

    md("## 5 · Detect CUDA (halts if unavailable)"),
    code(
        "%cd /content/aeon_bundle\n"
        "!python scripts/colab/env_check.py\n"
    ),

    md("## 6 · Benchmark Aeon on this GPU"),
    code(
        "%cd /content/aeon_bundle\n"
        "!python scripts/colab/benchmark.py --root . --tokens 50000\n"
    ),

    md("## 7 · Training-run parameters",
       "Set `DRY_RUN = True` to verify the pipeline without a long run.",
       "",
       "**Google Drive checkpoint dir.** Adjust `DRIVE_RUN_DIR` if you want a different Drive location; checkpoints are what survives session termination."),
    code(
        "DRY_RUN = True\n"
        "\n"
        "import os\n"
        "assert 'MYDRIVE' in globals(), (\n"
        "    'MYDRIVE is not defined. Run cell 1 (Mount Google Drive) first.')\n"
        "assert os.path.isdir(MYDRIVE), (\n"
        "    f'Drive mount lost; {MYDRIVE!r} no longer exists. Re-run cell 1.')\n"
        "\n"
        "DRIVE_RUN_DIR = os.path.join(MYDRIVE, 'aeon_fluency_run')\n"
        "STAGE1_CK_DIR = os.path.join(DRIVE_RUN_DIR, 'stage1_checkpoints')\n"
        "STAGE2_CK_DIR = os.path.join(DRIVE_RUN_DIR, 'stage2_checkpoints')\n"
        "\n"
        "STAGE1_TARGET_TOKENS  = 100_000_000\n"
        "STAGE2_TARGET_TOKENS  =  10_000_000\n"
        "CHECKPOINT_EVERY_TOK  =     250_000\n"
        "CHECKPOINT_EVERY_SEC  =       1_800  # 30 minutes\n"
        "\n"
        "# Set a wall-time cap per session so a single free-Colab session\n"
        "# always halts cleanly with a fresh checkpoint on Drive.\n"
        "SESSION_WALL_TIME_SEC = 42_000  # ~11.5 hours\n"
        "\n"
        "if DRY_RUN:\n"
        "    STAGE1_TARGET_TOKENS = 20_000\n"
        "    STAGE2_TARGET_TOKENS = 5_000\n"
        "    CHECKPOINT_EVERY_TOK = 5_000\n"
        "    CHECKPOINT_EVERY_SEC = 120\n"
        "    SESSION_WALL_TIME_SEC = 300\n"
        "\n"
        "import os\n"
        "os.makedirs(STAGE1_CK_DIR, exist_ok=True)\n"
        "os.makedirs(STAGE2_CK_DIR, exist_ok=True)\n"
        "print('DRY_RUN =', DRY_RUN)\n"
        "print('STAGE1_TARGET_TOKENS =', STAGE1_TARGET_TOKENS)\n"
        "print('STAGE2_TARGET_TOKENS =', STAGE2_TARGET_TOKENS)\n"
        "print('STAGE1_CK_DIR =', STAGE1_CK_DIR)\n"
        "print('STAGE2_CK_DIR =', STAGE2_CK_DIR)\n"
    ),

    md("## 8 · Stage 1 — train from protected P2 on WikiText-103 raw",
       "Resumable. Checkpoint frequency: every `CHECKPOINT_EVERY_TOK` tokens OR every `CHECKPOINT_EVERY_SEC` seconds, whichever fires first. Session halts cleanly at `SESSION_WALL_TIME_SEC` with the current checkpoint written; **the next Colab session will resume automatically** from the latest checkpoint on Drive.",
       "",
       "Re-run this cell as many times as needed until `STAGE1_TARGET_TOKENS` is reached."),
    code(
        "%cd /content/aeon_bundle\n"
        "PARENT = 'runs/aeon_lbc1_P2/final.pt'   # protected P2 (never overwritten)\n"
        "cmd = [\n"
        "    'python', 'scripts/colab/train_stage.py',\n"
        "    '--root', '.', '--stage', 'stage1',\n"
        "    '--parent', PARENT,\n"
        "    '--checkpoint-dir', STAGE1_CK_DIR,\n"
        "    '--target-tokens', str(STAGE1_TARGET_TOKENS),\n"
        "    '--checkpoint-every-tokens', str(CHECKPOINT_EVERY_TOK),\n"
        "    '--checkpoint-every-seconds', str(CHECKPOINT_EVERY_SEC),\n"
        "    '--wall-time-cap-seconds', str(SESSION_WALL_TIME_SEC),\n"
        "]\n"
        "if DRY_RUN: cmd.append('--dry-run')\n"
        "import subprocess\n"
        "subprocess.check_call(cmd)\n"
    ),

    md("## 9 · Stage 1 validation (WikiText valid — never test)"),
    code(
        "%cd /content/aeon_bundle\n"
        "import json, glob, os\n"
        "cks = sorted(glob.glob(f'{STAGE1_CK_DIR}/checkpoint_*.pt'))\n"
        "assert cks, 'No stage1 checkpoints yet — run cell 8 first.'\n"
        "latest = cks[-1]\n"
        "print('evaluating', latest)\n"
        "import subprocess\n"
        "subprocess.check_call(['python', 'scripts/colab/evaluate_and_generate.py',\n"
        "    '--root', '.', '--mode', 'stage1_valid',\n"
        "    '--checkpoint', latest,\n"
        "    '--out', f'{STAGE1_CK_DIR}/eval_stage1_valid_latest.json'])\n"
        "print(open(f'{STAGE1_CK_DIR}/eval_stage1_valid_latest.json').read())\n"
    ),

    md("## 10 · Stage 2 — Dolly-15k response-masked instruction tuning",
       "Uses the best Stage-1 checkpoint as parent. Excludes every retired ID and the locked `fresh_eval` subset from the training pool.",
       "",
       "Do not run this until Stage 1 has been trained to a reasonable validation loss (WikiText valid perplexity that Dylan judges acceptable)."),
    code(
        "%cd /content/aeon_bundle\n"
        "import glob\n"
        "cks = sorted(glob.glob(f'{STAGE1_CK_DIR}/checkpoint_*.pt'))\n"
        "assert cks, 'No stage1 checkpoints — cannot start stage2.'\n"
        "best_stage1 = cks[-1]  # Or: whatever checkpoint Dylan chose after cell 9\n"
        "print('stage2 parent =', best_stage1)\n"
        "cmd = [\n"
        "    'python', 'scripts/colab/train_stage.py',\n"
        "    '--root', '.', '--stage', 'stage2',\n"
        "    '--parent', best_stage1,\n"
        "    '--checkpoint-dir', STAGE2_CK_DIR,\n"
        "    '--target-tokens', str(STAGE2_TARGET_TOKENS),\n"
        "    '--checkpoint-every-tokens', str(CHECKPOINT_EVERY_TOK),\n"
        "    '--checkpoint-every-seconds', str(CHECKPOINT_EVERY_SEC),\n"
        "    '--wall-time-cap-seconds', str(SESSION_WALL_TIME_SEC),\n"
        "    '--seq-len', '256', '--batch-size', '4',\n"
        "]\n"
        "if DRY_RUN: cmd.append('--dry-run')\n"
        "import subprocess\n"
        "subprocess.check_call(cmd)\n"
    ),

    md("## 10b · Stage 2 checkpoint-selection signal (stage2_val — never a promotion gate)",
       "Locked at manifest time (`stage2_val_lock_sha256` re-verified before scoring). "
       "Use this loss between Stage-2 checkpoints to pick a stopping point. "
       "It is NOT a promotion signal; that is `fresh_eval`'s single-use role."),
    code(
        "%cd /content/aeon_bundle\n"
        "import glob\n"
        "cks = sorted(glob.glob(f'{STAGE2_CK_DIR}/checkpoint_*.pt'))\n"
        "assert cks, 'No stage2 checkpoints — run cell 10 first.'\n"
        "latest = cks[-1]\n"
        "import subprocess\n"
        "subprocess.check_call(['python', 'scripts/colab/evaluate_and_generate.py',\n"
        "    '--root', '.', '--mode', 'stage2_val',\n"
        "    '--checkpoint', latest,\n"
        "    '--out', f'{STAGE2_CK_DIR}/eval_stage2_val_latest.json'])\n"
        "print(open(f'{STAGE2_CK_DIR}/eval_stage2_val_latest.json').read())\n"
    ),

    md("## 11 · Stage 2 evaluation — fresh_eval only (single-use promotion gate)",
       "The evaluator verifies `fresh_eval_lock_sha256` before scoring; any drift aborts.",
       "",
       "This value is one input to Dylan's approval decision — not a substitute for it. Do not use it to pick between checkpoints; that is `stage2_val`'s role above."),
    code(
        "%cd /content/aeon_bundle\n"
        "import glob\n"
        "cks = sorted(glob.glob(f'{STAGE2_CK_DIR}/checkpoint_*.pt'))\n"
        "assert cks, 'No stage2 checkpoints — run cell 10 first.'\n"
        "latest = cks[-1]\n"
        "import subprocess\n"
        "subprocess.check_call(['python', 'scripts/colab/evaluate_and_generate.py',\n"
        "    '--root', '.', '--mode', 'stage2_fresh',\n"
        "    '--checkpoint', latest,\n"
        "    '--out', f'{STAGE2_CK_DIR}/eval_stage2_fresh_latest.json'])\n"
        "print(open(f'{STAGE2_CK_DIR}/eval_stage2_fresh_latest.json').read())\n"
    ),

    md("## 12 · Raw generations after a major checkpoint",
       "Deterministic greedy. Streamed decode is verified to equal one-shot decode. No rewriting. Dylan reviews these before any release approval."),
    code(
        "%cd /content/aeon_bundle\n"
        "import glob, os\n"
        "STAGE = 'stage2'  # switch to 'stage1' if you want Stage-1 samples\n"
        "ck_dir = STAGE2_CK_DIR if STAGE == 'stage2' else STAGE1_CK_DIR\n"
        "cks = sorted(glob.glob(f'{ck_dir}/checkpoint_*.pt'))\n"
        "assert cks, f'No {STAGE} checkpoints.'\n"
        "latest = cks[-1]\n"
        "out_path = f'{ck_dir}/raw_generations_{STAGE}_latest.json'\n"
        "import subprocess\n"
        "subprocess.check_call(['python', 'scripts/colab/evaluate_and_generate.py',\n"
        "    '--root', '.', '--mode', 'generate',\n"
        "    '--checkpoint', latest,\n"
        "    '--prompt-count', '25', '--max-new-tokens', '64',\n"
        "    '--out', out_path])\n"
        "print('wrote', out_path)\n"
    ),

    md("## Resume-later reminder",
       "If your Colab session times out, just:",
       "",
       "1. Reconnect to the same runtime (Runtime > Reconnect).",
       "2. Re-run cells 1 – 7 (fast: mount, unzip, verify, GPU check).",
       "3. Re-run cell 8 (Stage 1) — it resumes automatically from the latest checkpoint on Drive.",
       "4. Later, run cells 10 – 12 for Stage 2 and evaluation.",
       "",
       "The protected P2 checkpoint, tokenizer, architecture, K=16, margins, parameter count, and state-dict topology are preserved across every checkpoint and every session."),
]


def main() -> int:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python",
                            "name": "python3"},
            "language_info": {"name": "python", "version": "3.10"},
            "colab": {"provenance": []},
            "accelerator": "GPU",
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parent.parent.parent / "Aeon_English_Fluency_Colab.ipynb"
    out.write_text(json.dumps(nb, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(CELLS)} cells)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
