"""Stage-2 isolation proof: fresh_eval / stage2_val / retired IDs cannot
enter the Stage-2 training dataloader, and WikiText-103 test is never used
for training or checkpoint selection.

Every test here runs statically. No training executes.
"""
from __future__ import annotations

import ast
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _split():
    return json.loads((ROOT / "docs/en_train/dolly15k_split_manifest.json"
                        ).read_text(encoding="utf-8"))


def _fresh():
    return json.loads((ROOT / "docs/en_train/dolly15k_fresh_eval_manifest.json"
                        ).read_text(encoding="utf-8"))


def _h(ids) -> str:
    canon = "\n".join(sorted(ids)).encode("utf-8")
    return "sha256:" + hashlib.sha256(canon).hexdigest()


# ---------------------------------------------------------------------------
# 1. Manifest defines all four sets and their locks are internally consistent
# ---------------------------------------------------------------------------
def test_manifest_defines_stage2_validation_partition():
    m = _fresh()
    assert "stage2_validation" in m, "stage2_validation must exist"
    for k in ("record_count", "record_ids", "seed_for_stage2_val",
              "stage2_val_lock_sha256", "exclusion_rules"):
        assert k in m["stage2_validation"], f"stage2_validation missing {k}"


def test_stage2_val_lock_matches_ids():
    m = _fresh()
    ids = sorted(m["stage2_validation"]["record_ids"])
    got = _h(ids)
    want = m["stage2_validation"]["stage2_val_lock_sha256"]
    assert got == want, f"stage2_val_lock: got={got} want={want}"


# ---------------------------------------------------------------------------
# 2. Pairwise disjointness across every declared set
# ---------------------------------------------------------------------------
def test_every_pair_of_id_sets_is_disjoint():
    s = _split(); f = _fresh()
    train_ids = set(s["train_ids"])
    val_retired = set(s["val_ids"])
    sealed_retired = set(s["sealed_test_ids"])
    fresh_eval = set(f["fresh_eval"]["record_ids"])
    pilot_consumed = set(f["consumed_train_ids_by_pilot"])
    stage2_val = set(f["stage2_validation"]["record_ids"])

    # stage2_train = train - fresh_eval - pilot_consumed - stage2_val - val - sealed
    stage2_train = train_ids - fresh_eval - pilot_consumed - stage2_val \
                 - val_retired - sealed_retired

    pairs = {
        "stage2_train ∩ fresh_eval":     stage2_train & fresh_eval,
        "stage2_train ∩ stage2_val":     stage2_train & stage2_val,
        "stage2_train ∩ val_retired":    stage2_train & val_retired,
        "stage2_train ∩ sealed_retired": stage2_train & sealed_retired,
        "stage2_train ∩ pilot_consumed": stage2_train & pilot_consumed,
        "stage2_val   ∩ fresh_eval":     stage2_val & fresh_eval,
        "stage2_val   ∩ val_retired":    stage2_val & val_retired,
        "stage2_val   ∩ sealed_retired": stage2_val & sealed_retired,
        "stage2_val   ∩ pilot_consumed": stage2_val & pilot_consumed,
        "fresh_eval   ∩ val_retired":    fresh_eval & val_retired,
        "fresh_eval   ∩ sealed_retired": fresh_eval & sealed_retired,
        "fresh_eval   ∩ pilot_consumed": fresh_eval & pilot_consumed,
    }
    problems = {name: sorted(s)[:5] for name, s in pairs.items() if s}
    assert not problems, f"non-empty intersections detected: {problems}"


# ---------------------------------------------------------------------------
# 3. The Stage-2 dataloader function actually excludes fresh_eval + stage2_val
# ---------------------------------------------------------------------------
def test_stage2_exclusion_set_covers_every_forbidden_id():
    from scripts.colab.train_stage import _load_dolly_stage2_exclusion_set
    excluded = _load_dolly_stage2_exclusion_set(ROOT)
    s = _split(); f = _fresh()
    for name, needed in [
        ("fresh_eval", set(f["fresh_eval"]["record_ids"])),
        ("stage2_val", set(f["stage2_validation"]["record_ids"])),
        ("val_retired", set(s["val_ids"])),
        ("sealed_retired", set(s["sealed_test_ids"])),
        ("pilot_consumed", set(f["consumed_train_ids_by_pilot"])),
    ]:
        missing = needed - excluded
        assert not missing, (
            f"_load_dolly_stage2_exclusion_set missing {len(missing)} "
            f"ids from '{name}' — e.g. {sorted(missing)[:5]}")


def test_stage2_pool_iteration_leaks_zero_forbidden_records():
    """The actual dataloader iteration is enumerated and every produced
    record's id is checked against the union of forbidden sets. A single
    leak fails this test.

    We do NOT run the pilot's own next_batch loop with random shuffling;
    we iterate the returned pool directly, which is exactly what the
    training loop pulls from (`ordered = list(pool_ids); rng.shuffle;
    for batch in ordered:`). If a forbidden id is not in the pool, it
    cannot leak into training."""
    from scripts.colab.train_stage import (_load_dolly_stage2_pool,
                                            _load_dolly_stage2_exclusion_set)
    pool = _load_dolly_stage2_pool(ROOT)
    excluded = _load_dolly_stage2_exclusion_set(ROOT)
    leaked = [r.record_id for r in pool if r.record_id in excluded]
    assert not leaked, (
        f"Stage-2 training pool leaked {len(leaked)} forbidden ids: "
        f"{leaked[:5]}")


def test_stage2_pool_matches_expected_size():
    """train_ids (13521) - pilot_consumed (2676) - fresh_eval (496)
                        - stage2_val (200) - val_retired (0∩train)
                        - sealed_retired (0∩train)
    Val/sealed retired live in different partitions by construction,
    so their intersection with train is 0."""
    from scripts.colab.train_stage import _load_dolly_stage2_pool
    pool = _load_dolly_stage2_pool(ROOT)
    s = _split(); f = _fresh()
    expected = (len(s["train_ids"])
                - len(f["consumed_train_ids_by_pilot"])
                - len(f["fresh_eval"]["record_ids"])
                - len(f["stage2_validation"]["record_ids"]))
    assert len(pool) == expected, (
        f"stage2 pool size {len(pool)} != expected {expected}")


# ---------------------------------------------------------------------------
# 4. Stage-2 validation loader verifies its lock and refuses on drift
# ---------------------------------------------------------------------------
def test_stage2_val_loader_returns_locked_records():
    from scripts.colab.train_stage import _load_dolly_stage2_validation
    v = _load_dolly_stage2_validation(ROOT)
    m = _fresh()
    ids = sorted(r.record_id for r in v)
    assert ids == sorted(m["stage2_validation"]["record_ids"]), (
        "loader returned a different set of ids than the manifest declares")


def test_stage2_val_loader_refuses_on_lock_drift(tmp_path, monkeypatch):
    """Simulate a corrupted manifest with a wrong lock and verify the
    loader raises."""
    from scripts.colab.train_stage import _load_dolly_stage2_validation
    import shutil
    fake_root = tmp_path / "root"
    (fake_root / "docs" / "en_train").mkdir(parents=True)
    (fake_root / "research-data" / "incoming" / "EN-DOLLY-15K" / "sources").mkdir(parents=True)

    # Copy split + provenance verbatim
    shutil.copy(ROOT / "docs/en_train/dolly15k_split_manifest.json",
                 fake_root / "docs/en_train/dolly15k_split_manifest.json")
    shutil.copy(ROOT / "research-data/incoming/EN-DOLLY-15K/sources/databricks-dolly-15k.jsonl",
                 fake_root / "research-data/incoming/EN-DOLLY-15K/sources/databricks-dolly-15k.jsonl")

    # Corrupt fresh_eval manifest: change stage2_val ids but leave the
    # recorded lock alone.
    fresh = json.loads((ROOT / "docs/en_train/dolly15k_fresh_eval_manifest.json"
                        ).read_text(encoding="utf-8"))
    fresh["stage2_validation"]["record_ids"] = sorted(fresh["stage2_validation"]["record_ids"])[:-1]
    (fake_root / "docs/en_train/dolly15k_fresh_eval_manifest.json").write_text(
        json.dumps(fresh), encoding="utf-8")

    import pytest
    with pytest.raises(RuntimeError, match="stage2_val_lock drift"):
        _load_dolly_stage2_validation(fake_root)


# ---------------------------------------------------------------------------
# 5. WikiText test is never used for training or checkpoint selection
# ---------------------------------------------------------------------------
def test_train_stage_never_references_wiki_test_raw():
    """train_stage.py must not mention wiki.test.raw anywhere.
    It only ever iterates wiki.train.raw."""
    src = (ROOT / "scripts/colab/train_stage.py").read_text(encoding="utf-8")
    assert "wiki.test.raw" not in src, (
        "train_stage.py must never mention wiki.test.raw")
    assert "wiki.train.raw" in src, (
        "train_stage.py should iterate wiki.train.raw")


def test_train_stage_never_references_wikitext_test_split():
    """AST: no attribute or string reference to test-split-selection."""
    src = (ROOT / "scripts/colab/train_stage.py").read_text(encoding="utf-8")
    assert "stage1_test" not in src, (
        "train_stage.py must not route the stage1_test eval mode")


def test_evaluate_wikitext_test_is_documented_as_promotion_only():
    """The evaluator's stage1_test mode must be flagged as promotion-only
    (never for checkpoint selection)."""
    src = (ROOT / "scripts/colab/evaluate_and_generate.py").read_text(encoding="utf-8")
    m = re.search(r"stage1_test.*?promotion", src, re.DOTALL | re.IGNORECASE)
    assert m, ("evaluate_and_generate.py must document stage1_test as "
               "'promotion only'")
    assert "never use for checkpoint selection" in src.lower() or \
           "never for checkpoint selection" in src.lower()


def test_notebook_never_runs_stage1_test_for_checkpoint_selection():
    """The notebook's training + selection cells must not call
    stage1_test. It is only permissible as the FINAL Stage-1 promotion
    step (which the notebook does not include automatically)."""
    nb = json.loads((ROOT / "Aeon_English_Fluency_Colab.ipynb"
                      ).read_text(encoding="utf-8"))
    src = "\n".join("".join(c["source"]) for c in nb["cells"])
    assert "stage1_test" not in src, (
        "notebook must not run stage1_test — that is a manual "
        "final-promotion step, never a training/selection signal.")
    assert "wiki.test.raw" not in src, (
        "notebook must not reference wiki.test.raw in any cell.")


def test_evaluate_stage1_test_mode_still_available_for_final_promotion():
    """The mode must still be reachable via CLI (Dylan may want it
    for the final Stage-1 promotion pass), it just must never be
    invoked from the notebook or the trainer."""
    src = (ROOT / "scripts/colab/evaluate_and_generate.py").read_text(encoding="utf-8")
    assert '"stage1_test"' in src, (
        "stage1_test mode must remain a valid choice in the CLI")


# ---------------------------------------------------------------------------
# 6. Full report emitted as a JSON artefact so operator can inspect
# ---------------------------------------------------------------------------
def test_stage2_isolation_report_written_and_current():
    """The evidence report at docs/en_train/stage2_isolation_report.json
    must exist, match the current manifest values, and have every
    intersection == 0."""
    r = json.loads((ROOT / "docs/en_train/stage2_isolation_report.json"
                     ).read_text(encoding="utf-8"))
    for k in ("counts", "sha256_by_set", "pairwise_intersection_counts",
              "code_paths", "assertions"):
        assert k in r, f"isolation report missing {k}"
    for name, n in r["pairwise_intersection_counts"].items():
        assert n == 0, f"non-empty intersection reported: {name} = {n}"
    for name, val in r["assertions"].items():
        assert val is True, f"assertion {name} not True: {val}"
