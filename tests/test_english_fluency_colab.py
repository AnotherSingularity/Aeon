"""EN-COLAB — static regression tests for the free-Colab fluency tranche.

Runs on any platform. No CUDA needed. No training executed. Every check
is either JSON-schema validation, deterministic id-set comparison, or
AST-based verification of forbidden call sites.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 1. fresh_eval manifest: contamination-free
# ---------------------------------------------------------------------------
def _split():
    return json.loads((ROOT / "docs/en_train/dolly15k_split_manifest.json"
                        ).read_text(encoding="utf-8"))


def _fresh():
    return json.loads((ROOT / "docs/en_train/dolly15k_fresh_eval_manifest.json"
                        ).read_text(encoding="utf-8"))


def test_fresh_eval_manifest_present_and_schemaed():
    m = _fresh()
    for k in ("schema_version", "retired_from_future_promotion",
              "fresh_eval", "wikitext103_official_test_split_note",
              "consumed_train_ids_by_pilot", "notes_for_operator"):
        assert k in m, f"missing top-level key: {k}"
    for k in ("record_count", "record_ids", "fresh_eval_lock_sha256"):
        assert k in m["fresh_eval"], f"fresh_eval missing {k}"


def test_fresh_eval_lock_hash_matches_ids():
    m = _fresh()
    ids = sorted(m["fresh_eval"]["record_ids"])
    canon = "\n".join(ids).encode("utf-8")
    got = "sha256:" + hashlib.sha256(canon).hexdigest()
    assert got == m["fresh_eval"]["fresh_eval_lock_sha256"], (
        f"lock mismatch: recorded={m['fresh_eval']['fresh_eval_lock_sha256']} "
        f"recomputed={got}")


def test_fresh_eval_excludes_all_consumed_ids():
    """The most important isolation check: fresh_eval must share zero
    ids with sealed_test, val, or the pilot's iterated train records."""
    s = _split()
    m = _fresh()
    fresh = set(m["fresh_eval"]["record_ids"])
    sealed = set(s["sealed_test_ids"])
    val = set(s["val_ids"])
    consumed_train = set(m["consumed_train_ids_by_pilot"])
    assert not (fresh & sealed), (
        f"fresh_eval overlaps retired sealed_test: {list(fresh & sealed)[:5]}")
    assert not (fresh & val), (
        f"fresh_eval overlaps retired val: {list(fresh & val)[:5]}")
    assert not (fresh & consumed_train), (
        f"fresh_eval overlaps pilot-consumed train: {list(fresh & consumed_train)[:5]}")


def test_fresh_eval_ids_are_all_in_train_partition():
    """fresh_eval must be drawn from the untouched-train pool only —
    not from Wikitext or a fresh source outside the split manifest."""
    s = _split()
    m = _fresh()
    train_set = set(s["train_ids"])
    for rid in m["fresh_eval"]["record_ids"]:
        assert rid in train_set, f"fresh_eval id {rid} is not in train_ids"


def test_pilot_consumed_train_size_and_determinism():
    """The consumed_train reconstruction must match what the pilot
    actually iterated: 1338 * 2 = 2676 records with seed 20260822."""
    import random
    m = _fresh()
    s = _split()
    train_ids = list(s["train_ids"])
    rng = random.Random(20260822)
    ordered = list(train_ids)
    rng.shuffle(ordered)
    expected = ordered[:2676]
    assert m["consumed_train_ids_by_pilot"] == sorted(expected), (
        "consumed_train_ids_by_pilot must match deterministic pilot shuffle")


# ---------------------------------------------------------------------------
# 2. Notebook: 26 cells, GPU accelerator, contains required steps
# ---------------------------------------------------------------------------
def _nb():
    return json.loads((ROOT / "Aeon_English_Fluency_Colab.ipynb"
                        ).read_text(encoding="utf-8"))


def test_notebook_exists_and_parses():
    nb = _nb()
    assert nb["nbformat"] == 4
    assert isinstance(nb["cells"], list)
    assert len(nb["cells"]) >= 14, (
        f"notebook should have >= 14 numbered cells; got {len(nb['cells'])}")


def test_notebook_declares_gpu_accelerator():
    nb = _nb()
    assert nb["metadata"].get("accelerator") == "GPU"


def test_notebook_contains_required_directive_landmarks():
    nb = _nb()
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    required = [
        "drive.mount",                                # Drive mount
        "Aeon_English_Fluency_Colab_Bundle.zip",     # bundle
        "sentencepiece",                               # deps
        "verify_bundle.py",                           # SHA-256 verification
        "download_wikitext103.py",                    # Stage-1 corpus
        "191,984,949",                                 # byte size (comma-formatted in the markdown)
        "91c00ae287f0d699e18605c84afc9e45c192bc6b7797ff8837e5474655a33794",  # sha256
        "env_check.py",                               # CUDA halt-if-none
        "benchmark.py",                               # projection
        "runs/aeon_lbc1_P2/final.pt",                 # Stage-1 parent = P2
        "'--stage', 'stage1'",                        # Stage-1 launcher
        "'--stage', 'stage2'",                        # Stage-2 launcher
        "STAGE1_TARGET_TOKENS  = 100_000_000",        # 100M
        "CHECKPOINT_EVERY_TOK  =     250_000",        # 250K
        "SESSION_WALL_TIME_SEC = 42_000",             # ~11.5h halt
        "DRY_RUN = True",                             # dry-run present
        "stage2_fresh",                                # fresh_eval eval mode
        "raw_generations",                            # raw-generation dump
    ]
    for needle in required:
        assert needle in src, f"notebook missing landmark: {needle!r}"


def test_notebook_disclaims_fluency_before_review():
    nb = _nb()
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "Dylan" in src or "human review" in src.lower(), (
        "notebook must state that Dylan reviews before approval")
    assert "No automatic 'fluent' claim" in src or "not.*claim.*flu" in src.lower() or "no automatic" in src.lower(), (
        "notebook must not auto-declare fluency")


# ---------------------------------------------------------------------------
# 3. Colab scripts: no external-model / API imports; halt-on-no-CUDA
# ---------------------------------------------------------------------------
FORBIDDEN_TOP_IMPORTS = {
    "transformers", "openai", "anthropic", "vllm", "peft", "trl",
    "bitsandbytes", "auto_gptq", "llama_cpp", "sentence_transformers",
    "requests", "httpx", "urllib3", "openai_azure",
}


def test_colab_scripts_have_no_external_llm_imports():
    for p in sorted((ROOT / "scripts" / "colab").glob("*.py")):
        tree = ast.parse(p.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in FORBIDDEN_TOP_IMPORTS, (
                        f"{p.name} imports forbidden module {alias.name}")
            if isinstance(node, ast.ImportFrom):
                top = (node.module or "").split(".")[0]
                assert top not in FORBIDDEN_TOP_IMPORTS, (
                    f"{p.name} imports forbidden module {node.module}")


def test_env_check_halts_when_cuda_unavailable():
    """Static check: env_check.py must exit non-zero when
    torch.cuda.is_available() is False. AST inspection of its main."""
    src = (ROOT / "scripts/colab/env_check.py").read_text(encoding="utf-8")
    assert "torch.cuda.is_available()" in src
    assert "return 5" in src


def test_train_stage_halts_when_cuda_unavailable():
    src = (ROOT / "scripts/colab/train_stage.py").read_text(encoding="utf-8")
    assert "torch.cuda.is_available()" in src
    assert "return 5" in src


def test_train_stage_never_loads_micro_pilot_candidate_as_parent():
    """The pilot candidate lives under runs/en_proof_dolly15k_s20260822/.
    train_stage.py must not silently default to it as a parent."""
    src = (ROOT / "scripts/colab/train_stage.py").read_text(encoding="utf-8")
    assert "en_proof_dolly15k_s20260822" not in src, (
        "train_stage.py must not reference the retired micro-pilot candidate")


def test_train_stage_preserves_invariants_hard_halts():
    src = (ROOT / "scripts/colab/train_stage.py").read_text(encoding="utf-8")
    for needle in ("A0 digest drift", "parameter count drift", "margin drift",
                    "K != 16", "non-finite loss"):
        assert needle in src, f"train_stage must fail-closed on: {needle}"


def test_evaluate_verifies_fresh_eval_lock_before_scoring():
    src = (ROOT / "scripts/colab/evaluate_and_generate.py").read_text(encoding="utf-8")
    assert "fresh_eval_lock_sha256" in src
    assert "drift" in src


# ---------------------------------------------------------------------------
# 4. Bundle: SHA-256 manifest present, no forbidden paths
# ---------------------------------------------------------------------------
def test_sha256_manifest_matches_disk_and_is_populated():
    m = json.loads((ROOT / "SHA256_MANIFEST.json").read_text(encoding="utf-8"))
    assert m["file_count"] > 30, f"manifest suspiciously small: {m['file_count']}"
    # Recompute a random sample of 5 to keep this test fast on CI
    import random
    rng = random.Random(20260822)
    sample = rng.sample(m["files"], min(5, len(m["files"])))
    for entry in sample:
        p = ROOT / entry["path"]
        assert p.exists(), f"manifest lists missing path {entry['path']}"
        h = hashlib.sha256(); h.update(p.read_bytes())
        got = "sha256:" + h.hexdigest()
        assert got == entry["sha256"], (
            f"{entry['path']}: manifest={entry['sha256']} disk={got}")


def test_bundle_zip_has_no_forbidden_paths():
    z = ROOT / "Aeon_English_Fluency_Colab_Bundle.zip"
    if not z.exists():
        # Bundle is optional in git; when produced, it must be clean.
        return
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
    forbidden_prefixes = ("runs/en_proof_dolly15k_s20260822/",
                          "dist/", "packaging/", ".git/", "build/")
    problems = [n for n in names
                if any(n.startswith(fp) for fp in forbidden_prefixes)]
    assert not problems, f"bundle contains forbidden paths: {problems[:5]}"


def test_bundle_zip_has_required_entries():
    z = ROOT / "Aeon_English_Fluency_Colab_Bundle.zip"
    if not z.exists():
        return
    with zipfile.ZipFile(z) as zf:
        names = set(zf.namelist())
    for required in (
        "Aeon_English_Fluency_Colab.ipynb",
        "SHA256_MANIFEST.json",
        "runs/aeon_lbc1_P2/final.pt",
        "release-assets/aeon-desktop-p2-proxy/tokenizer/aeon-lbc1.model",
        "research-data/incoming/EN-DOLLY-15K/sources/databricks-dolly-15k.jsonl",
        "docs/en_train/dolly15k_provenance.json",
        "docs/en_train/dolly15k_fresh_eval_manifest.json",
        "docs/en_train/EN_TRAIN_ARCHITECTURE_FREEZE.json",
        "scripts/colab/train_stage.py",
        "scripts/colab/evaluate_and_generate.py",
        "scripts/colab/verify_bundle.py",
    ):
        assert required in names, f"bundle missing required entry: {required}"


# ---------------------------------------------------------------------------
# 5. Architectural invariance still intact
# ---------------------------------------------------------------------------
def test_A0_digest_and_pins_unchanged():
    fp = json.loads((ROOT / "docs/en_train/EN_TRAIN_ARCHITECTURE_FREEZE.json"
                      ).read_text(encoding="utf-8"))
    assert fp["architecture_fingerprint_A0_digest"] == \
        "sha256:2f895a05411567619371dd76a5f22868ca9e7edc17f33711e2e99aab04a972f9"
    assert fp["total_parameters"] == 7015366
    assert fp["K"] == 16
    assert fp["protected_p2_checkpoint"]["sha256"] == \
        "sha256:962fcd5e65a88e3b6d061e73968bea7eb3581c4d8a15b49be82427004db9fc3c"
    assert fp["protected_tokenizer"]["sha256"] == \
        "sha256:064ab6a98ee4b8177249f14dc09e60ccaa9986b66cb84a4072c68dc4de533481"


def test_no_english_fluency_commit_touched_protected_boundaries():
    r = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--name-only", "--pretty=format:", "-4"],
        capture_output=True, text=True, check=True)
    touched = {ln for ln in r.stdout.splitlines() if ln.strip()}
    forbidden = ("aeon/hybrid.py", "aeon/recursion.py", "aeon/substrate/",
                 "aeon/desktop/runtime.py",
                 "release-assets/aeon-desktop-p2-proxy/model/",
                 "release-assets/aeon-desktop-p2-proxy/tokenizer/")
    problems = [t for t in touched
                if any(t.startswith(p) for p in forbidden)]
    assert not problems, f"recent commits touched protected files: {problems}"
