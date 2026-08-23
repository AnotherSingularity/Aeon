"""scripts/colab/build_bundle.py — assemble Aeon_English_Fluency_Colab_Bundle.zip.

Bundle contents (per the ENGLISH_FLUENCY_COLAB directive Section §Bundle):
  1. Aeon_English_Fluency_Colab.ipynb
  2. aeon/ source tree required to train Aeon
  3. Protected P2 checkpoint (runs/aeon_lbc1_P2/final.pt)
  4. Protected tokenizer (release-assets/.../tokenizer/aeon-lbc1.model)
  5. Dolly-15K raw JSONL already verified in the sandbox
  6. Dataset provenance + license files
  7. Architecture-invariance manifest (EN_TRAIN_ARCHITECTURE_FREEZE.json)
  8. Resumable-training configuration (bundled colab scripts + configs)
  9. Evaluation infrastructure (scripts/colab/*)
 10. SHA256_MANIFEST.json (SHA-256 of every included file)

Explicit NOT-included: Windows build outputs, installer files, unrelated
corpora, the micro-pilot candidate, credentials, tokens, secrets, .git.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path


def _sha256(path: Path, buf: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(buf), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# What goes in the bundle. Order matters only for readability.
#
# The aeon.* section is the *complete runtime closure* reachable from the
# notebook's entry points, verified by
# scripts/colab/build_bundle.py::_verify_runtime_closure() at build time.
# It includes every module actually imported when
# scripts.colab.train_stage._build_model_and_tokenizer runs against the
# extracted bundle, PLUS the lazy shuttle+bypass modules that
# HybridModel.forward can pull in through its optional observer/shuttle
# args. Do not shrink this list without regenerating the runtime-closure
# proof; missing modules do not fail INCLUDE — they fail Colab at import.
INCLUDE = [
    # 1. Notebook
    "Aeon_English_Fluency_Colab.ipynb",

    # 2. Aeon source tree — the full runtime closure
    "aeon/__init__.py",
    "aeon/hybrid.py",
    "aeon/recursion.py",
    "aeon/transformer.py",
    "aeon/tokenizer.py",
    "aeon/desktop/protocol.py",  # for AttributionSettings compatibility

    # 2a. substrate package — every file (imports are cross-referential)
    "aeon/substrate/__init__.py",
    "aeon/substrate/port.py",
    "aeon/substrate/conformance.py",
    "aeon/substrate/matrix_cell.py",
    "aeon/substrate/vector_cell.py",
    "aeon/substrate/feedback.py",

    # 2b. bypass package — pulled in by HybridModel.forward when observer!=None
    "aeon/bypass/__init__.py",
    "aeon/bypass/signal_trace.py",

    # 2c. shuttle package — pulled in by HybridModel.forward when shuttle!=None,
    #     and __init__.py exports FIXED_K referenced by test_windows_packaging.
    #     Include every shuttle file so no lazy import can miss.
    "aeon/shuttle/__init__.py",
    "aeon/shuttle/routing.py",
    "aeon/shuttle/audit.py",
    "aeon/shuttle/broadcast.py",
    "aeon/shuttle/contracts.py",
    "aeon/shuttle/lease.py",
    "aeon/shuttle/lifecycle.py",
    "aeon/shuttle/policy.py",
    "aeon/shuttle/backpressure.py",

    # 2d. en_train package
    "aeon/en_train/__init__.py",
    "aeon/en_train/losses.py",
    "aeon/en_train/proof.py",
    "aeon/en_train/proof_harness.py",
    "aeon/en_train/proof_pilot.py",
    "aeon/en_train/dolly_split.py",
    "aeon/en_train/trainer.py",
    "aeon/en_train/data.py",
    "aeon/en_train/eval.py",
    "aeon/en_train/attribution.py",

    # 3. Protected P2 checkpoint (never overwritten)
    "runs/aeon_lbc1_P2/final.pt",

    # 4. Protected tokenizer + release architecture manifests
    "release-assets/aeon-desktop-p2-proxy/tokenizer/aeon-lbc1.model",
    "release-assets/aeon-desktop-p2-proxy/manifests/architecture_manifest.json",
    "release-assets/aeon-desktop-p2-proxy/manifests/release_manifest.json",
    "release-assets/aeon-desktop-p2-proxy/config/aeon_lbc1_proxy.yaml",

    # 5. Dolly-15k raw JSONL (verified in this session)
    "research-data/incoming/EN-DOLLY-15K/sources/databricks-dolly-15k.jsonl",

    # 6. Dataset provenance + license files
    "docs/en_train/dolly15k_provenance.json",
    "docs/en_train/dolly15k_split_manifest.json",
    "docs/en_train/dolly15k_fresh_eval_manifest.json",
    "docs/en_train/stage2_isolation_report.json",

    # 7. Architecture-invariance manifests
    "docs/en_train/EN_TRAIN_ARCHITECTURE_FREEZE.json",
    "docs/en_train/en_train_repository_symbol_mapping.json",
    "docs/en_train/en_train_clock_mapping.json",

    # 8+9. Colab scripts (resumable training + evaluation + verification)
    "scripts/colab/verify_bundle.py",
    "scripts/colab/download_wikitext103.py",
    "scripts/colab/env_check.py",
    "scripts/colab/benchmark.py",
    "scripts/colab/train_stage.py",
    "scripts/colab/evaluate_and_generate.py",
    "scripts/colab/README.md",

    # Extra doc glue for context
    "docs/en_train/ENGLISH_PROOF_PROTOCOL.md",
    "docs/en_train/EN_TRAIN_CORRECTED_MATHEMATICAL_SPEC.md",
]

# Optional configs — include if present. NOTE: substrate/* moved to
# INCLUDE (required) because they are runtime-mandatory. Nothing here
# may be a runtime import; only truly optional configs.
OPTIONAL_INCLUDE = [
    "configs/latent_bypass/aeon_lbc1_proxy.yaml",
    "configs/aeon_v1.yaml",
]

# Explicitly excluded (regex-like prefixes)
FORBIDDEN_PREFIXES = (
    "runs/en_proof_dolly15k_s20260822/",   # micro-pilot candidate
    "dist/", "build/",                      # windows/build artefacts
    "packaging/", ".build-venv/",           # windows packaging
    ".git/", ".github/",
    "release-assets/aeon-desktop-p2-proxy/model/",  # duplicate of P2 in a different form
)


def _validate_no_forbidden(paths):
    for p in paths:
        for fp in FORBIDDEN_PREFIXES:
            if p.startswith(fp):
                raise RuntimeError(f"forbidden path in bundle: {p}")


def _walk_include(root: Path):
    """Yield (rel_path, abs_path) tuples for INCLUDE + OPTIONAL_INCLUDE that exist."""
    for rel in INCLUDE:
        full = root / rel
        if not full.exists():
            raise FileNotFoundError(f"required bundle file missing: {rel}")
        yield rel, full
    for rel in OPTIONAL_INCLUDE:
        full = root / rel
        if full.exists():
            yield rel, full


def _verify_runtime_closure(zip_path: Path) -> dict:
    """Extract the built zip into a temp dir and spawn a python
    subprocess that (1) sets PYTHONPATH to the extracted dir ONLY,
    (2) unsets the repository's cwd from the module search path,
    (3) imports HybridModel + every substrate module + colab helper
    entry points, and (4) instantiates a tiny HybridModel through
    _build_model_and_tokenizer against the extracted bundle.
    Any missing module fails the build."""
    import subprocess, sys, tempfile, zipfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(td)
        # Clean subprocess env: PYTHONPATH = extracted dir only
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = str(td)
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        probe = td / "_closure_probe.py"
        probe.write_text(
            "import sys, os\n"
            "# The extracted-bundle root MUST be the only place aeon comes from.\n"
            "sys.path = [p for p in sys.path if p and 'AeonV0.02' not in p]\n"
            "import aeon\n"
            "assert 'AeonV0.02' not in aeon.__file__, aeon.__file__\n"
            "# Force the full runtime closure the notebook triggers:\n"
            "from aeon.hybrid import HybridModel\n"
            "from aeon.transformer import AeonTransformerConfig\n"
            "from aeon.tokenizer import AeonTokenizer\n"
            "from aeon.substrate import make_substrate\n"
            "from aeon.substrate import port, conformance, matrix_cell, vector_cell, feedback\n"
            "from aeon.bypass import signal_trace\n"
            "from aeon.shuttle import routing\n"
            "import aeon.en_train.losses, aeon.en_train.proof, aeon.en_train.proof_pilot\n"
            "import aeon.en_train.dolly_split, aeon.en_train.proof_harness\n"
            "import aeon.en_train.trainer, aeon.en_train.data, aeon.en_train.eval\n"
            "import aeon.en_train.attribution\n"
            "# Assert every loaded aeon.* module resolves under the bundle root\n"
            "root = os.path.dirname(aeon.__file__)\n"
            "for m in list(sys.modules):\n"
            "    if not m.startswith('aeon'): continue\n"
            "    mod = sys.modules[m]\n"
            "    f = getattr(mod, '__file__', None)\n"
            "    if f is None: continue\n"
            "    assert os.path.commonpath([f, root]).startswith(os.path.dirname(root)), \\\n"
            "        f'{m} loaded from {f}, not under {root}'\n"
            "# Now build the model + tokenizer through the actual helper\n"
            "sys.path.insert(0, os.getcwd())  # scripts/colab is under cwd\n"
            "from scripts.colab.train_stage import _build_model_and_tokenizer\n"
            "from pathlib import Path\n"
            "model, tok = _build_model_and_tokenizer(Path(os.getcwd()))\n"
            "print('OK', 'params=', sum(p.numel() for p in model.parameters()),\n"
            "       'vocab=', tok.vocab_size)\n"
        )
        result = subprocess.run(
            [sys.executable, str(probe)],
            cwd=str(td), env=env, capture_output=True, text=True,
            timeout=180)
        ok = result.returncode == 0
        return {"ok": ok, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", default="Aeon_English_Fluency_Colab_Bundle.zip")
    ap.add_argument("--dry-run", action="store_true",
                    help="build the SHA-256 manifest only; do not write the zip")
    args = ap.parse_args()
    root = Path(args.root).resolve()

    files = list(_walk_include(root))
    _validate_no_forbidden([r for r, _ in files])

    # Compute SHA-256 for every file
    entries = []
    total_bytes = 0
    for rel, full in files:
        s = _sha256(full)
        n = full.stat().st_size
        total_bytes += n
        entries.append({"path": rel, "sha256": s, "bytes": n})
    entries.sort(key=lambda e: e["path"])

    manifest = {
        "schema_version": 1,
        "bundle_name": "Aeon_English_Fluency_Colab_Bundle",
        "produced_at_head": os.environ.get("GIT_HEAD", "TBD"),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "explicit_exclusions": {
            "windows_build_outputs": True,
            "installer_files": True,
            "micro_pilot_candidate": True,
            "credentials_tokens_secrets": True,
            "git_metadata": True,
            "unrelated_corpora": True,
        },
        "files": entries,
    }
    # Write manifest next to the bundle location for reference too
    manifest_disk = root / "SHA256_MANIFEST.json"
    manifest_disk.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    print(f"wrote {manifest_disk}")

    if args.dry_run:
        print(f"dry-run: manifest built, {len(entries)} files, "
              f"{total_bytes/1e6:.1f} MB total")
        return 0

    out = root / args.out
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for rel, full in files:
            zf.write(full, arcname=rel)
        zf.writestr("SHA256_MANIFEST.json",
                    json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    sz = out.stat().st_size
    sha = _sha256(out)
    print(f"wrote {out}")
    print(f"  size:   {sz:,} bytes ({sz/1e6:.1f} MB)")
    print(f"  sha256: {sha}")

    (root / (args.out + ".sha256")).write_text(f"{sha}  {args.out}\n",
                                                encoding="ascii")

    # Build-time runtime closure guard: extract the just-written zip
    # into a temp dir and prove HybridModel + every substrate module +
    # helper entry points import cleanly from it, with the repo cwd
    # scrubbed from PYTHONPATH. Any missing module fails the build.
    print("[bundle] runtime-closure guard: extracting + probing…")
    closure = _verify_runtime_closure(out)
    if not closure["ok"]:
        print("STDOUT:", closure["stdout"])
        print("STDERR:", closure["stderr"], file=sys.stderr)
        raise RuntimeError(
            "runtime-closure guard failed: the extracted bundle cannot "
            "import HybridModel or its dependencies from a clean env. "
            "Fix INCLUDE and rerun.")
    print(f"[bundle] runtime-closure guard: {closure['stdout'].strip()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
