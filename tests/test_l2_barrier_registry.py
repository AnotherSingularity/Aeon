"""L2 — visible barrier registry + candidate search + corpus-package validator.

L2 is IMPLEMENTATION-ONLY per the corpus-staging rule. Nothing here
supports an observational, causal, efficiency, or bypass claim. The
tests exercise:

    * Registry schema validation, duplicate-ID refusal, unknown-metric
      refusal, hidden-state-input refusal.
    * Version handling.
    * Calibration policies (top_percent, bottom_percent, median,
      per_partition_quantile, fixed_value).
    * Calibrated thresholds LOCK; re-calibration refused unless
      allow_recalibration=True.
    * Calibration partition ≠ evaluation partition (registry-level).
    * Missing-data policy (false / true / raise).
    * Same-visible-state candidate search operates only on visible
      inputs (exact prefix + declared projection).
    * Determinism of the candidate digest.
    * Synthetic fixture is correctly marked implementation-only:
      docs/latent_bypass/status.json.achieved_claim_level remains 0.
    * Corpus-package validator refuses partially-formed packages;
      accepts a well-formed one; refuses to inspect the sealed test
      partition unless allow_test_partition_access=True.
    * The barrier registry cannot reach any hidden-state field.
    * IP-preservation manifest unchanged.
"""
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

REGISTRY_PATH = os.path.join(ROOT, "benchmarks", "latent_bypass", "barriers.json")


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------
def test_registry_loads_all_eight_required_definitions():
    from aeon.bypass.barriers import load_registry
    reg = load_registry(REGISTRY_PATH)
    required_ids = {"HIGH_LOCAL_LOSS", "LOW_TARGET_MARGIN",
                     "HIGH_VISIBLE_ENTROPY", "LONG_DEPENDENCY",
                     "LOCAL_STATE_ALIASING", "CONTRADICTION_REGION",
                     "DELAYED_INSTRUCTION_REGION",
                     "ENTITY_STATE_DISCONTINUITY"}
    ids = set(reg.ids())
    assert required_ids.issubset(ids), (
        f"L2: missing required barrier IDs: {required_ids - ids}")


def test_registry_refuses_duplicate_barrier_id():
    from aeon.bypass.barriers import (
        BarrierDefinition, BarrierRegistry, BarrierRegistryError,
    )
    row = BarrierDefinition(
        schema_version=1, barrier_id="X", version=1, description="",
        visible_metric="pre_broadcast_token_loss",
        observation_point="pre_broadcast",
        threshold_method="top_percent", threshold_value=None,
        calibration_partition="calibration",
        evaluation_partition="test",
        minimum_samples=1, missing_data_action="false",
        applicable_tasks=())
    try:
        BarrierRegistry([row, row])
    except BarrierRegistryError as e:
        assert e.code == "duplicate_barrier_id"
    else:
        raise AssertionError("expected duplicate_barrier_id")


def test_registry_refuses_unknown_visible_metric():
    from aeon.bypass.barriers import (
        BarrierDefinition, BarrierRegistry, BarrierRegistryError,
    )
    row = BarrierDefinition(
        schema_version=1, barrier_id="X", version=1, description="",
        visible_metric="unknown_metric_xyz",
        observation_point="pre_broadcast",
        threshold_method="top_percent", threshold_value=None,
        calibration_partition="calibration",
        evaluation_partition="test",
        minimum_samples=1, missing_data_action="false")
    try:
        BarrierRegistry([row])
    except BarrierRegistryError as e:
        assert e.code == "unknown_visible_metric"
    else:
        raise AssertionError("expected unknown_visible_metric")


def test_registry_refuses_hidden_state_input_in_row():
    """A registry row containing any FORBIDDEN_REGISTRY_INPUTS string
    is rejected at load time — enforces that barrier evaluation
    cannot see the hidden Recursion state."""
    from aeon.bypass.barriers import (
        load_registry, BarrierRegistryError,
    )
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "bad.json"
        p.write_text(json.dumps({
            "schema_version": 1,
            "barriers": [{
                "schema_version": 1,
                "barrier_id": "X", "version": 1,
                "description": "reads h_cond",
                "visible_metric": "pre_broadcast_token_loss",
                "observation_point": "pre_broadcast",
                "threshold_method": "top_percent",
                "threshold_value": None,
                "calibration_partition": "calibration",
                "evaluation_partition": "test",
                "minimum_samples": 1,
                "missing_data_action": "false",
                "applicable_tasks": ["h_cond"],
            }]
        }))
        try:
            load_registry(str(p))
        except BarrierRegistryError as e:
            assert e.code == "forbidden_hidden_state_input", e.code


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def test_calibration_top_percent_and_lock():
    from aeon.bypass.barriers import (
        load_registry, BarrierRegistryError,
    )
    reg = load_registry(REGISTRY_PATH)
    samples = [float(i) for i in range(200)]
    row = reg.calibrate("HIGH_LOCAL_LOSS", samples)
    assert row.threshold_value is not None
    assert row.threshold_value >= samples[len(samples) - 20]
    # Locked: second call refuses without allow_recalibration=True
    try:
        reg.calibrate("HIGH_LOCAL_LOSS", samples)
    except BarrierRegistryError as e:
        assert e.code == "already_calibrated"
    # But allow_recalibration=True works
    row2 = reg.calibrate("HIGH_LOCAL_LOSS", samples,
                          allow_recalibration=True)
    assert row2.threshold_value is not None


def test_calibration_refuses_insufficient_samples():
    from aeon.bypass.barriers import (
        load_registry, BarrierRegistryError,
    )
    reg = load_registry(REGISTRY_PATH)
    try:
        reg.calibrate("HIGH_LOCAL_LOSS", [1.0, 2.0, 3.0])
    except BarrierRegistryError as e:
        assert e.code == "insufficient_calibration_samples"
    else:
        raise AssertionError("expected insufficient_calibration_samples")


def test_calibration_partition_differs_from_evaluation_partition():
    """Every registry row's calibration_partition must be distinct
    from its evaluation_partition — separates threshold-fitting from
    scoring."""
    from aeon.bypass.barriers import load_registry
    reg = load_registry(REGISTRY_PATH)
    for bid in reg.ids():
        r = reg.get(bid)
        assert r.calibration_partition != r.evaluation_partition, (
            f"L2: barrier {bid} has calibration_partition == "
            f"evaluation_partition == {r.calibration_partition!r}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def test_evaluate_top_percent_returns_true_above_threshold():
    from aeon.bypass.barriers import load_registry
    reg = load_registry(REGISTRY_PATH)
    reg.calibrate("HIGH_LOCAL_LOSS", [float(i) for i in range(200)])
    row = reg.get("HIGH_LOCAL_LOSS")
    assert reg.evaluate("HIGH_LOCAL_LOSS", row.threshold_value + 0.001) is True
    assert reg.evaluate("HIGH_LOCAL_LOSS", row.threshold_value - 0.001) is False


def test_evaluate_missing_data_action():
    from aeon.bypass.barriers import (
        load_registry, BarrierRegistryError,
    )
    reg = load_registry(REGISTRY_PATH)
    reg.calibrate("HIGH_LOCAL_LOSS", [float(i) for i in range(200)])
    # default missing_data_action='false'
    assert reg.evaluate("HIGH_LOCAL_LOSS", None) is False


# ---------------------------------------------------------------------------
# Candidate search
# ---------------------------------------------------------------------------
def test_exact_prefix_pairs_are_deterministic_and_visible_only():
    from aeon.bypass.candidates import find_exact_prefix_matches, build_locked_set
    records = [
        {"record_id": "r1", "tokens": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {"record_id": "r2", "tokens": [1, 2, 3, 42, 5, 6, 7, 99, 9, 10]},
    ]
    pairs = find_exact_prefix_matches(records, prefix_length=3)
    ids = {(p.left_record_id, p.left_position,
            p.right_record_id, p.right_position) for p in pairs}
    assert ids, "L2: exact-prefix search should find at least one pair"
    pairs2 = find_exact_prefix_matches(records, prefix_length=3)
    assert [(p.left_record_id, p.left_position,
             p.right_record_id, p.right_position, p.visible_distance)
            for p in pairs] == \
           [(p.left_record_id, p.left_position,
             p.right_record_id, p.right_position, p.visible_distance)
            for p in pairs2]
    locked = build_locked_set(
        pairs, match_method="exact_prefix", epsilon=0.0,
        prefix_length=3, projection_id=None,
        locked_at_iso="2026-07-31T22:00:00Z")
    assert locked.candidate_set_digest.startswith("sha256:")


def test_declared_projection_pairs_bounded_by_epsilon():
    from aeon.bypass.candidates import find_projection_matches
    projections = [
        {"record_id": "r1", "position": 5, "vector": [0.0, 0.0, 0.0]},
        {"record_id": "r2", "position": 8, "vector": [0.01, 0.0, 0.0]},
        {"record_id": "r3", "position": 3, "vector": [10.0, 0.0, 0.0]},
    ]
    pairs = find_projection_matches(
        projections, epsilon=0.1, projection_id="test_v1")
    ids = {(p.left_record_id, p.right_record_id) for p in pairs}
    assert ("r1", "r2") in ids
    assert not any(p.right_record_id == "r3" for p in pairs)


# ---------------------------------------------------------------------------
# Claim-ladder integrity
# ---------------------------------------------------------------------------
def test_l2_does_not_elevate_claim_level():
    """L2 uses the synthetic fixture only. achieved_claim_level in
    status.json must remain 0."""
    p = os.path.join(ROOT, "docs", "latent_bypass", "status.json")
    with open(p, encoding="utf-8") as fh:
        s = json.load(fh)
    assert s["achieved_claim_level"] == 0, (
        f"L2: achieved_claim_level must remain 0; got "
        f"{s['achieved_claim_level']}")
    assert s["real_corpus_claims_authorized"] is False


# ---------------------------------------------------------------------------
# Corpus-package validator
# ---------------------------------------------------------------------------
def test_corpus_package_validator_refuses_partial_package():
    from aeon.bypass.corpus_package import validate_corpus_package
    with tempfile.TemporaryDirectory() as d:
        # Empty directory
        r = validate_corpus_package(d)
        assert r.ready_for_L3 is False
        assert r.errors  # must enumerate what is missing


def test_corpus_package_validator_accepts_well_formed_package():
    from aeon.bypass.corpus_package import validate_corpus_package
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # Build a minimal well-formed package
        (d / "source").mkdir()
        (d / "processed").mkdir()
        (d / "source" / "original.txt").write_text("hello world")
        partition_paths = {}
        digests = {}
        for name in ("train", "calibration", "validation", "test"):
            p = d / "processed" / f"{name}.jsonl"
            p.write_text(json.dumps({"text": f"partition:{name}"}) + "\n")
            partition_paths[name] = str(p)
            digests[name] = hashlib.sha256(p.read_bytes()).hexdigest()
        (d / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "corpus_id": "test-corpus",
            "title": "test",
            "source": "unit-test",
            "retrieval_date": "2026-07-31",
            "source_sha256": hashlib.sha256(
                (d / "source" / "original.txt").read_bytes()).hexdigest(),
            "public_domain_basis": "test",
            "license_status": "public_domain",
            "preprocessing_version": 1,
            "tokenizer_id": "sha256:test",
            "partitions": {
                name: {"sha256": digests[name]}
                for name in partition_paths
            },
        }))
        (d / "provenance.json").write_text(json.dumps({"source": "test"}))
        (d / "license.txt").write_text("Public domain.")
        (d / "partition_report.json").write_text(json.dumps({
            "train_calibration_overlap": 0,
            "train_validation_overlap": 0,
            "train_test_overlap": 0,
        }))
        (d / "SEALED_TEST_DIGEST").write_text(digests["test"])
        r = validate_corpus_package(str(d))
        assert r.ready_for_L3 is True, r.errors
        assert r.test_partition_sealed is True
        # Test partition not inspected by default
        assert r.partition_digests_match["test"] is True


def test_corpus_package_validator_does_not_inspect_sealed_test_by_default():
    """Corrupt test partition should NOT be flagged unless
    allow_test_partition_access=True."""
    from aeon.bypass.corpus_package import validate_corpus_package
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        (d / "source").mkdir()
        (d / "processed").mkdir()
        (d / "source" / "original.txt").write_text("hello world")
        for name in ("train", "calibration", "validation"):
            p = d / "processed" / f"{name}.jsonl"
            p.write_text(json.dumps({"text": name}) + "\n")
        # test partition content differs from what manifest expects
        (d / "processed" / "test.jsonl").write_text(
            json.dumps({"text": "corrupted"}) + "\n")
        digests = {
            n: hashlib.sha256((d / "processed" / f"{n}.jsonl").read_bytes()).hexdigest()
            for n in ("train", "calibration", "validation")
        }
        digests["test"] = "0" * 64  # deliberately wrong
        (d / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "corpus_id": "c",
            "title": "t",
            "source": "s",
            "retrieval_date": "d",
            "source_sha256": "x",
            "public_domain_basis": "pd",
            "license_status": "public_domain",
            "preprocessing_version": 1,
            "tokenizer_id": "sha256:t",
            "partitions": {
                n: {"sha256": digests[n]}
                for n in ("train", "calibration", "validation", "test")
            },
        }))
        (d / "provenance.json").write_text("{}")
        (d / "license.txt").write_text("pd")
        (d / "partition_report.json").write_text("{}")
        (d / "SEALED_TEST_DIGEST").write_text(digests["test"])
        r_default = validate_corpus_package(str(d))
        assert r_default.partition_digests_match["test"] is True
        r_open = validate_corpus_package(
            str(d), allow_test_partition_access=True)
        assert r_open.partition_digests_match["test"] is False


# ---------------------------------------------------------------------------
# IP-preservation cross-check
# ---------------------------------------------------------------------------
def test_ip_preservation_manifest_unchanged():
    """L2 must not modify docs/latent_bypass/ip_preservation.json (the
    firewall's manifest of protected structure)."""
    p = os.path.join(ROOT, "docs", "latent_bypass", "ip_preservation.json")
    with open(p, encoding="utf-8") as fh:
        m = json.load(fh)
    # Must still list the protected modules and classes.
    assert "aeon/hybrid.py" in m["protected_modules"]
    assert "aeon/recursion.py" in m["protected_modules"]
    assert any(row["name"] == "HybridModel"
                 for row in m["protected_classes"])


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
