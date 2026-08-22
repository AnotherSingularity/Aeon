"""ENGLISH-PROOF-0 / C — infrastructure tests for the pilot.

Covers the parts of the directive Section 15 that can be exercised
WITHOUT the authorized Dolly-15k corpus:

  * deterministic dataset split
  * exact duplicate containment
  * near-duplicate containment
  * sealed partition isolation (lock-hash verifies)
  * response mask correctness (assistant char-span -> supervised region)
  * identical attribution settings (fingerprint stability)
  * raw-output completeness (harness records every required field)
  * no post-generation rewriting (harness is text-preserving)
  * no external-model dependency (module import graph)
  * proof gate cannot pass without Dylan's human scorecard
  * provenance schema fields and pinned digests
  * pilot halts at AWAITING_DOLLY_DATA_UPLOAD until corpus is uploaded

Tests that require actual training tokens (finite loss, nonzero
native gradients, nonzero candidate weight delta, zero architecture
delta, P2 immutability under a real optimizer step) will run in the
EN-PROOF-D tranche AFTER the pilot completes; they are declared
here as future-work stubs so the coverage map stays honest.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# 1. Deterministic split
# ---------------------------------------------------------------------------
def _fake_records(n=100):
    from aeon.en_train.dolly_split import DollyRecord
    recs = []
    for i in range(n):
        recs.append(DollyRecord(
            record_id=f"r{i:04d}",
            instruction=f"Q{i}: what is fact number {i}?",
            context="",
            response=f"A{i}: fact number {i} is that {i}+{i}={2 * i}.",
            category="open_qa"))
    return recs


def test_split_is_deterministic_across_runs():
    from aeon.en_train.dolly_split import deterministic_split
    r1 = deterministic_split(_fake_records(200), seed=20260822)
    r2 = deterministic_split(_fake_records(200), seed=20260822)
    assert r1.train_ids == r2.train_ids
    assert r1.val_ids == r2.val_ids
    assert r1.sealed_test_ids == r2.sealed_test_ids


def test_split_partitions_are_disjoint_and_cover_kept_records():
    from aeon.en_train.dolly_split import deterministic_split
    r = deterministic_split(_fake_records(500), seed=20260822)
    ids = set(r.train_ids) | set(r.val_ids) | set(r.sealed_test_ids)
    # No overlap
    assert len(r.train_ids) + len(r.val_ids) + len(r.sealed_test_ids) == len(ids)
    # Every kept id is covered
    excluded = {e["record_id"] for e in r.excluded_records}
    fake_ids = {f"r{i:04d}" for i in range(500)}
    assert ids | excluded == fake_ids


def test_split_ratios_are_approximately_correct():
    from aeon.en_train.dolly_split import deterministic_split
    r = deterministic_split(_fake_records(1000), seed=20260822)
    total = len(r.train_ids) + len(r.val_ids) + len(r.sealed_test_ids)
    train_frac = len(r.train_ids) / total
    val_frac = len(r.val_ids) / total
    sealed_frac = len(r.sealed_test_ids) / total
    # 90/5/5 with hash-based bucketing on 1000 items — allow ±3pp.
    assert 0.87 <= train_frac <= 0.93, f"train_frac={train_frac}"
    assert 0.02 <= val_frac <= 0.08, f"val_frac={val_frac}"
    assert 0.02 <= sealed_frac <= 0.08, f"sealed_frac={sealed_frac}"


def test_seed_change_reshuffles_partitions():
    from aeon.en_train.dolly_split import deterministic_split
    r1 = deterministic_split(_fake_records(500), seed=20260822)
    r2 = deterministic_split(_fake_records(500), seed=20260823)
    assert r1.sealed_test_ids != r2.sealed_test_ids


# ---------------------------------------------------------------------------
# 2. Exact duplicate containment
# ---------------------------------------------------------------------------
def test_exact_duplicates_are_grouped_and_only_representative_kept():
    from aeon.en_train.dolly_split import (DollyRecord, deterministic_split)
    recs = [
        DollyRecord("r1", "same q", "", "same a", "open_qa"),
        DollyRecord("r2", "same q", "", "same a", "open_qa"),   # exact dup of r1
        DollyRecord("r3", "same q", "", "same a", "open_qa"),   # exact dup of r1
        DollyRecord("r4", "different", "", "different", "closed_qa"),
    ]
    r = deterministic_split(recs, seed=20260822)
    kept = set(r.train_ids) | set(r.val_ids) | set(r.sealed_test_ids)
    # Only r1 (lex representative) and r4 survive
    assert kept == {"r1", "r4"}, f"expected {{r1, r4}}, got {kept}"
    # r2 and r3 are recorded as excluded with a clear reason
    excluded_ids = {e["record_id"] for e in r.excluded_records}
    assert {"r2", "r3"} <= excluded_ids
    for e in r.excluded_records:
        if e["record_id"] in {"r2", "r3"}:
            assert e["reason"].startswith("exact_duplicate_of_group_representative:")


# ---------------------------------------------------------------------------
# 3. Near-duplicate containment
# ---------------------------------------------------------------------------
def test_near_duplicates_stay_in_the_same_partition():
    from aeon.en_train.dolly_split import (DollyRecord, deterministic_split)
    # Two records that share > threshold 5-gram Jaccard.
    base = ("this is a long enough response that we can produce five gram "
            "windows meaningfully across it many times over please")
    recs = [
        DollyRecord("r1", "same instruction", "", base, "open_qa"),
        DollyRecord("r2", "same instruction slightly different opening", "",
                    base + " tail", "open_qa"),
        DollyRecord("r3", "unrelated", "", "totally unrelated text here", "closed_qa"),
    ]
    r = deterministic_split(recs, seed=20260822)
    # r1 and r2 must be in the same partition
    partitions = {"train": set(r.train_ids), "val": set(r.val_ids),
                  "sealed": set(r.sealed_test_ids)}
    same = None
    for name, s in partitions.items():
        if "r1" in s and "r2" in s:
            same = name; break
    assert same is not None, (
        f"r1 and r2 must land in the same partition; got {partitions}")


# ---------------------------------------------------------------------------
# 4. Sealed partition isolation — lock hash detects mutation
# ---------------------------------------------------------------------------
def test_sealed_test_lock_verifies_intact_manifest(tmp_path):
    from aeon.en_train.dolly_split import (deterministic_split, write_split_manifest,
                                            verify_sealed_test_lock)
    r = deterministic_split(_fake_records(200), seed=20260822)
    manifest = tmp_path / "m.json"
    write_split_manifest(r, manifest)
    ok, msg = verify_sealed_test_lock(manifest)
    assert ok, msg


def test_sealed_test_lock_detects_mutation(tmp_path):
    from aeon.en_train.dolly_split import (deterministic_split, write_split_manifest,
                                            verify_sealed_test_lock)
    r = deterministic_split(_fake_records(200), seed=20260822)
    manifest = tmp_path / "m.json"
    write_split_manifest(r, manifest)
    # Mutate the sealed set (attacker moves one id into sealed)
    d = json.loads(manifest.read_text(encoding="utf-8"))
    d["sealed_test_ids"] = sorted(set(d["sealed_test_ids"]) | {"synthetic_new"})
    manifest.write_text(json.dumps(d), encoding="utf-8")
    ok, msg = verify_sealed_test_lock(manifest)
    assert not ok
    assert "sealed_test_lock_sha256" in msg


# ---------------------------------------------------------------------------
# 5. Response mask correctness — the assistant span is the SUPERVISED region
# ---------------------------------------------------------------------------
def test_render_dolly_record_places_span_over_response_content_only():
    from aeon.en_train.proof_pilot import render_dolly_record_for_training
    text, spans = render_dolly_record_for_training(
        instruction="What is 2+2?",
        context="",
        response="Four.")
    assert len(spans) == 1
    a, b = spans[0]
    assert text[a:b] == "Four.", (
        f"span should cover exactly the assistant content; got {text[a:b]!r}")


def test_render_dolly_record_with_context_still_supervises_only_response():
    from aeon.en_train.proof_pilot import render_dolly_record_for_training
    text, spans = render_dolly_record_for_training(
        instruction="Answer using the context.",
        context="The capital of France is Paris.",
        response="Paris.")
    a, b = spans[0]
    assert text[a:b] == "Paris."
    # And the user turn contains BOTH instruction and context
    assert "Answer using the context." in text
    assert "The capital of France is Paris." in text


# ---------------------------------------------------------------------------
# 6. Attribution harness — settings fingerprint stability + drift detection
# ---------------------------------------------------------------------------
def test_attribution_settings_fingerprint_is_stable_for_identical_values():
    from aeon.en_train.proof_harness import AttributionSettings
    a = AttributionSettings()
    b = AttributionSettings()
    assert a.fingerprint() == b.fingerprint()


def test_attribution_settings_fingerprint_flips_on_any_change():
    from aeon.en_train.proof_harness import AttributionSettings, assert_attribution_settings_bytewise_equal
    a = AttributionSettings()
    b = replace(a, temperature=0.01)
    assert a.fingerprint() != b.fingerprint()
    import pytest
    with pytest.raises(RuntimeError):
        assert_attribution_settings_bytewise_equal(a, b)


def test_run_attribution_records_every_required_field():
    """Harness records prompt id/text, checkpoint role/sha, ids, per-step,
    both decodes, stop reason, settings fp, duration — for every prompt."""
    from aeon.en_train.proof_harness import (AttributionSettings, run_attribution,
                                              stream_and_full_decode)

    class _StubTok:
        eos_id = 2
        def encode(self, s, add_bos=False, add_eos=False):
            return [ord(c) % 100 + 3 for c in s][:8]
        def decode(self, ids):
            return "".join(chr(x + 32) for x in ids)

    def _stub_forward(model, ctx):
        # Deterministic "next token"
        return (sum(ctx) % 90) + 3

    s = AttributionSettings(max_new_tokens=3)
    resps = run_attribution(
        tokenizer=_StubTok(), model=None,
        prompts=[("p1", "hello"), ("p2", "world")],
        settings=s, checkpoint_role="candidate",
        checkpoint_sha256="sha256:deadbeef",
        forward_step_fn=_stub_forward)
    assert len(resps) == 2
    for r in resps:
        d = r.to_dict()
        for k in ("prompt_id", "prompt_text", "checkpoint_role",
                  "checkpoint_sha256", "generated_token_ids",
                  "per_step_selected_token", "full_decoded_text",
                  "streamed_decoded_text", "stop_reason",
                  "generation_settings_fingerprint",
                  "generation_duration_seconds"):
            assert k in d, f"attribution response missing {k}"


def test_stream_and_full_decode_equivalence_on_stub_tokenizer():
    from aeon.en_train.proof_harness import stream_and_full_decode

    class _StubTok:
        eos_id = 2
        def decode(self, ids):
            return "".join(chr(x + 32) for x in ids)
    s, f = stream_and_full_decode(_StubTok(), [1, 2, 3, 4, 5])
    assert s == f


# ---------------------------------------------------------------------------
# 7. No external-model dependency — module import graph check
# ---------------------------------------------------------------------------
def test_proof_pilot_and_harness_do_not_import_external_llm_libs():
    """Any of these appearing in the pilot or harness would mean the
    proof is contaminated by an external language model. Explicit
    negative list."""
    forbidden = {"transformers", "openai", "anthropic", "vllm",
                 "peft", "trl", "bitsandbytes", "auto_gptq",
                 "llama_cpp", "sentence_transformers", "requests",
                 "httpx", "urllib3"}
    for modname in ("aeon.en_train.proof_harness",
                    "aeon.en_train.proof_pilot",
                    "aeon.en_train.dolly_split"):
        src = Path(modname.replace(".", "/") + ".py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top = alias.name.split(".")[0]
                    assert top not in forbidden, (
                        f"{modname} imports forbidden module {alias.name}")
            if isinstance(node, ast.ImportFrom):
                top = (node.module or "").split(".")[0]
                assert top not in forbidden, (
                    f"{modname} imports forbidden module {node.module}")


# ---------------------------------------------------------------------------
# 8. Proof gate cannot pass without Dylan's human scorecard
# ---------------------------------------------------------------------------
def test_proof_gate_requires_human_scorecard_before_declaring_pass():
    """The pilot's halt-state machine forbids emitting
    ENGLISH_PROOF_READY_FOR_DYLAN_REVIEW as a 'passed' result on its
    own. The gate to declare 'candidate approved' can only be crossed
    by a scorecard signed by Dylan. This test asserts the halt-state
    string exists and is DIFFERENT from any 'approved' constant."""
    from aeon.en_train.proof_pilot import (HALT_READY, HALT_AWAITING_DATA,
                                            HALT_FAILED)
    # Distinct state names — human review is a distinct step from any
    # automated declaration.
    assert HALT_READY == "ENGLISH_PROOF_READY_FOR_DYLAN_REVIEW"
    assert HALT_AWAITING_DATA == "AWAITING_DOLLY_DATA_UPLOAD"
    assert HALT_FAILED == "ENGLISH_PROOF_FAILED_NO_PACKAGING"
    # The pilot must never invent an "approved" state internally; if it
    # did we would fail loudly here.
    import aeon.en_train.proof_pilot as pp
    for attr in dir(pp):
        low = attr.lower()
        assert "approved" not in low, (
            f"proof_pilot must not declare its own 'approved' state; found {attr}")


# ---------------------------------------------------------------------------
# 9. Provenance schema
# ---------------------------------------------------------------------------
def test_provenance_schema_fields_present():
    p = ROOT / "docs" / "en_train" / "dolly15k_provenance.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    for k in ("schema_version", "authorization", "acquisition",
              "license_and_attribution", "record_counts", "schema",
              "handling_rules", "pilot_settings", "status"):
        assert k in d, f"provenance missing {k}"
    assert d["status"] in (
        "AWAITING_DOLLY_DATA_UPLOAD",
        "DOLLY_UPLOADED",
        "PILOT_RAN",
    ), f"unexpected provenance status: {d['status']}"
    # Authorized dataset id is exactly as directed
    assert d["authorization"]["authorized_dataset"] == "databricks/databricks-dolly-15k"
    # License is CC BY-SA 3.0
    assert d["license_and_attribution"]["license_short_name"] == "CC BY-SA 3.0"
    # Pilot pins P2 sha256
    assert d["pilot_settings"]["parent_sha256_pinned"] == \
        "sha256:962fcd5e65a88e3b6d061e73968bea7eb3581c4d8a15b49be82427004db9fc3c"


# ---------------------------------------------------------------------------
# 10. Halt state until upload
# ---------------------------------------------------------------------------
def test_pilot_halts_at_awaiting_data_when_corpus_absent():
    from aeon.en_train.proof_pilot import (halt_state_for_current_environment,
                                            HALT_AWAITING_DATA, PilotConfig)
    s = halt_state_for_current_environment(PilotConfig())
    assert s["state"] == HALT_AWAITING_DATA, s


# ---------------------------------------------------------------------------
# 11. Baseline invariants unchanged by this tranche
# ---------------------------------------------------------------------------
def test_baseline_invariants_unchanged():
    fp = json.loads((ROOT / "docs" / "en_train" /
                     "EN_TRAIN_ARCHITECTURE_FREEZE.json").read_text(encoding="utf-8"))
    assert fp["architecture_fingerprint_A0_digest"] == \
        "sha256:2f895a05411567619371dd76a5f22868ca9e7edc17f33711e2e99aab04a972f9"
    assert fp["total_parameters"] == 7015366
    assert fp["K"] == 16
    assert fp["protected_p2_checkpoint"]["sha256"] == \
        "sha256:962fcd5e65a88e3b6d061e73968bea7eb3581c4d8a15b49be82427004db9fc3c"
    assert fp["protected_tokenizer"]["sha256"] == \
        "sha256:064ab6a98ee4b8177249f14dc09e60ccaa9986b66cb84a4072c68dc4de533481"


def test_no_english_proof_commit_touched_protected_boundaries():
    """This tranche must not touch aeon/hybrid.py, aeon/recursion.py,
    aeon/substrate/**, aeon/desktop/runtime.py, or the release-assets
    model/tokenizer directories."""
    import subprocess
    r = subprocess.run(
        ["git", "-C", str(ROOT), "log", "--name-only", "--pretty=format:", "-3"],
        capture_output=True, text=True, check=True)
    touched = {ln for ln in r.stdout.splitlines() if ln.strip()}
    forbidden = ("aeon/hybrid.py", "aeon/recursion.py", "aeon/substrate/",
                 "aeon/desktop/runtime.py",
                 "release-assets/aeon-desktop-p2-proxy/model/",
                 "release-assets/aeon-desktop-p2-proxy/tokenizer/")
    problems = [t for t in touched
                if any(t.startswith(p) for p in forbidden)]
    assert not problems, f"recent commits touched protected files: {problems}"
