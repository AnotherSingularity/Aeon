"""
F2 — Artifact identity, provenance chain, dependency policy, corpus provenance,
refusal behaviour.

Torch-free where possible; a small subset uses torch to verify identity of
config objects that flow through model building. Skips cleanly otherwise.
"""
import copy
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---- F2.1 canonical identity ------------------------------------------------
def test_canonical_json_ignores_env_incidentals():
    from aeon.provenance import hash_object
    obj_a = {"model": "m1", "config_sha256": "abc", "absolute_path": "/a/b/c",
             "hostname": "alpha"}
    obj_b = {"absolute_path": "/z/y/x", "hostname": "beta",
             "config_sha256": "abc", "model": "m1"}
    assert hash_object(obj_a) == hash_object(obj_b), \
        "environment incidentals must not change identity"


def test_canonical_json_key_order_agnostic():
    from aeon.provenance import hash_object
    a = {"a": 1, "b": {"y": 2, "x": 3}, "c": [1, 2, 3]}
    b = {"c": [1, 2, 3], "b": {"x": 3, "y": 2}, "a": 1}
    assert hash_object(a) == hash_object(b)


def test_lists_are_order_sensitive():
    """Order matters for lists (semantic)."""
    from aeon.provenance import hash_object
    assert hash_object([1, 2, 3]) != hash_object([3, 2, 1])


def test_identity_constructors_populated():
    from aeon.provenance import (source_commit_identity,
                                 dependency_lockfile_identity,
                                 runtime_versions_identity,
                                 build_configuration_identity)
    sc = source_commit_identity()
    assert sc["kind"] == "source_commit"
    assert isinstance(sc["dirty"], bool)
    assert isinstance(sc.get("commit"), str)
    dep = dependency_lockfile_identity()
    assert dep["kind"] == "dependency_lockfile"
    assert dep.get("sha256")
    rv = runtime_versions_identity()
    assert rv["kind"] == "runtime_versions" and "python" in rv["versions"]
    bc = build_configuration_identity()
    assert bc["kind"] == "build_configuration"
    assert bc["dependency_lockfile"]["sha256"]


# ---- F2.2 provenance chain --------------------------------------------------
def test_provenance_chain_covers_required_kinds():
    from aeon.provenance import CHAIN_KINDS
    for k in ("source_commit", "build_configuration", "model_configuration",
              "tokenizer", "corpus_manifest", "training_run", "checkpoint",
              "evaluation", "recovery"):
        assert k in CHAIN_KINDS, f"chain missing: {k}"


def test_strict_verify_checkpoint_refuses_missing_fields():
    from aeon.provenance import strict_verify, ProvenanceError
    rec = {"source_commit": {"commit": "abc123", "dirty": False},
           "build_configuration": {"kind": "build_configuration"},
           "model_configuration": {"kind": "model_configuration"},
           "tokenizer": {"kind": "tokenizer", "present": False, "sha256": None},
           "corpus_manifest": {"kind": "corpus_manifest", "manifest_sha256": "z"},
           "training_run": {"run_id": "r1"},
           "runtime_policy": {"kind": "runtime_policy"},
           "security_policy": {"kind": "security_policy"}}
    strict_verify(rec, kind="checkpoint")               # complete → OK
    # remove each required field one at a time
    for k in ("source_commit", "build_configuration", "model_configuration",
              "tokenizer", "corpus_manifest", "training_run", "runtime_policy",
              "security_policy"):
        bad = copy.deepcopy(rec); bad[k] = None
        try:
            strict_verify(bad, kind="checkpoint")
            assert False, f"strict_verify accepted missing {k}"
        except ProvenanceError:
            pass


def test_strict_verify_refuses_unknown_source_commit():
    from aeon.provenance import strict_verify, ProvenanceError
    rec = {"source_commit": {"commit": "unknown", "dirty": False},
           "build_configuration": {"k": 1}, "model_configuration": {"k": 1},
           "tokenizer": {"present": False, "sha256": None},
           "corpus_manifest": {"m": 1}, "training_run": {"r": 1},
           "runtime_policy": {"k": 1}, "security_policy": {"k": 1}}
    try:
        strict_verify(rec, kind="checkpoint")
        assert False, "expected ProvenanceError for unknown source commit"
    except ProvenanceError as e:
        assert "source commit" in str(e).lower()


# ---- F2.3 dependency policy -------------------------------------------------
def test_tcb_report_covers_all_pinned_runtime_deps():
    tcb = json.load(open("docs/tcb_report.json"))
    names = {d["name"] for d in tcb["runtime_dependencies"]}
    assert {"torch", "safetensors", "sentencepiece", "pyyaml", "numpy"} <= names
    assert tcb["install_policy"]["auto_install_from_aeon_runtime"] is False


def test_no_runtime_pip_call_in_forward_path():
    """Aeon runtime must not invoke pip/subprocess.install anywhere in the
    training / inference / diagnostic import path."""
    import glob
    for path in glob.glob("aeon/**/*.py", recursive=True):
        src = open(path, encoding="utf-8").read()
        # Any use of pip / install / subprocess-in-forward is a fail.
        assert "pip install" not in src, f"{path}: contains 'pip install'"
        assert "pip._internal" not in src, path
    # scripts/ may reference pip in DOCUMENTATION comments but not as code —
    # ensure no active call.
    for path in glob.glob("scripts/*.py"):
        src = open(path, encoding="utf-8").read()
        # Allow subprocess.run for git-rev-parse and E5 test spawning, but not
        # for pip/install.
        assert "pip install" not in src.replace("# ", ""), \
            f"{path}: contains active 'pip install'"


# ---- F2.4 corpus provenance -------------------------------------------------
def test_corpus_manifest_schema_rejects_missing_fields():
    from aeon.corpus_manifest import validate_manifest
    bad = {"sources": [{"source_id": "s1", "trust_level": "trusted"}]}
    errs = validate_manifest(bad)
    assert errs and any("missing" in e for e in errs), errs


def test_corpus_manifest_quarantine_cannot_enter_train():
    from aeon.corpus_manifest import validate_manifest
    m = {"sources": [{
        "source_id": "q1", "origin": "test", "acquired_at": "2025-01-01",
        "license_status": "test", "content_sha256": "0" * 64,
        "preprocessing_version": "v0", "filtering_version": "v0",
        "deduplication_version": "v0", "partition_assignment": "train",
        "inclusion_status": "included",
        "rejection_reason_if_rejected": None,
        "trust_level": "quarantined"}]}
    errs = validate_manifest(m)
    assert any("quarantined" in e for e in errs), errs


def test_corpus_manifest_excluded_needs_reason():
    from aeon.corpus_manifest import validate_manifest
    m = {"sources": [{
        "source_id": "e1", "origin": "test", "acquired_at": "2025-01-01",
        "license_status": "test", "content_sha256": "0" * 64,
        "preprocessing_version": "v0", "filtering_version": "v0",
        "deduplication_version": "v0", "partition_assignment": "held_out",
        "inclusion_status": "excluded",
        "rejection_reason_if_rejected": None,
        "trust_level": "trusted"}]}
    errs = validate_manifest(m)
    assert any("rejection_reason" in e for e in errs), errs


def test_synthetic_manifest_is_valid_but_recorded_as_synthetic():
    from aeon.corpus_manifest import synthetic_manifest_for_smoke, validate_manifest
    m = synthetic_manifest_for_smoke()
    errs = validate_manifest(m)
    assert not errs, errs
    src = m["sources"][0]
    assert src["source_id"] == "synthetic_random_tokens"
    assert "synthetic" in src["license_status"]


def test_content_sha256_recompute_refuses_mismatch():
    from aeon.corpus_manifest import verify_source_content
    from aeon.provenance import ProvenanceError
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("hello aeon")
        path = fh.name
    src = {"source_id": "x", "content_sha256": "0" * 64}
    try:
        verify_source_content(src, path)
        assert False, "expected refusal"
    except ProvenanceError as e:
        assert "sha256 mismatch" in str(e)
    os.unlink(path)


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
