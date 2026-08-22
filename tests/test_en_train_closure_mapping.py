"""EN-TRAIN-9 — Closure reference schema, mapping-contract, and
architectural-invariance tests.

Enforces the contract of docs/en_train/closure_reference_mapping.json
and validates the invariants required by the closure integration
tranche:

  1. PDF-provenance schema — required fields, valid classifications.
  2. Index binding — i, w, k are declared bookkeeping only; k is
     never mapped to an Aeon architectural clock.
  3. Prohibited-mapping ledger — Aeon slow clock cannot be equated
     with parameter update theta_{tau+1}.
  4. Aeon slow-clock cadence — exactly one RecursionJoiner.step call
     per K-window (ceil(T/K)).
  5. Aeon fast-clock cadence — exactly one substrate.step call per
     input token (T total).
  6. Renderer changes cannot be counted as learned English — the
     desktop runtime rendering path contains no writes to
     model.state_dict, no calls to model.parameters(), no gradient
     or optimizer references. AST-scoped.
  7. Protected checkpoint hash unchanged vs frozen fingerprint.
  8. Tokenizer hash unchanged vs frozen fingerprint.
  9. K remains exactly 16 in every consumer.
"""
from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLOSURE_MAPPING_PATH = ROOT / "docs" / "en_train" / "closure_reference_mapping.json"
SYMBOL_MAPPING_PATH = ROOT / "docs" / "en_train" / "en_train_repository_symbol_mapping.json"
ARCH_FREEZE_PATH = ROOT / "docs" / "en_train" / "EN_TRAIN_ARCHITECTURE_FREEZE.json"
RUNTIME_SOURCE = ROOT / "aeon" / "desktop" / "runtime.py"

VALID_CLASSIFICATIONS = {
    "implemented",
    "implemented_with_documentation_drift",
    "offline_training_only",
    "evaluation_only",
    "proposed_future_research",
    "prohibited_in_current_tranche",
    "descriptive_joint_state_only",
    # Chapter 10 rate 3 gets a joint label — treat both as valid.
    "offline_training_only OR evaluation_only",
    "implemented (offline learning) + evaluation_only (viability monitoring)",
    "implemented (mean-only) with a documented future-research consideration",
}


# ---------------------------------------------------------------------------
# 1. Schema — closure_reference_mapping.json has required fields
# ---------------------------------------------------------------------------
def _load_mapping():
    return json.loads(CLOSURE_MAPPING_PATH.read_text(encoding="utf-8"))


def test_closure_mapping_top_level_schema():
    m = _load_mapping()
    for field in ("schema_version", "purpose", "produced_at_head",
                  "reference", "classification_taxonomy",
                  "three_indices", "aeon_native_clocks_from_repository",
                  "dual_clock_collision_prohibition",
                  "descriptive_joint_state_only",
                  "three_rates_distinction", "pdf_concepts",
                  "consequences_for_current_tranche", "provenance"):
        assert field in m, f"closure_reference_mapping.json missing {field}"


def test_closure_mapping_reference_fields():
    m = _load_mapping()
    r = m["reference"]
    for field in ("filename", "sha256", "title", "author",
                  "page_count", "size_bytes", "role", "authority"):
        assert field in r, f"reference missing {field}"
    assert r["role"] == "research_reference"
    assert r["authority"].startswith("non-canonical")
    assert r["sha256"].startswith("sha256:") and len(r["sha256"]) == 71


def test_closure_mapping_taxonomy_matches_declared_set():
    m = _load_mapping()
    for c in m["classification_taxonomy"]:
        assert c in VALID_CLASSIFICATIONS - {  # exclude joint labels from strict set
            "offline_training_only OR evaluation_only",
            "implemented (offline learning) + evaluation_only (viability monitoring)",
            "implemented (mean-only) with a documented future-research consideration",
        }, f"taxonomy contains unknown classification: {c}"


def test_closure_mapping_pdf_concepts_have_required_fields():
    m = _load_mapping()
    required = {"concept", "pdf_location", "repository_counterpart",
                "implementation_status"}
    for i, c in enumerate(m["pdf_concepts"]):
        missing = required - set(c.keys())
        assert not missing, f"pdf_concepts[{i}] ({c.get('concept','?')}) missing {missing}"
        assert c["implementation_status"] in VALID_CLASSIFICATIONS, \
            f"pdf_concepts[{i}] has invalid status: {c['implementation_status']}"


# ---------------------------------------------------------------------------
# 2. Indices — i, w, k are bookkeeping only; k is not a clock
# ---------------------------------------------------------------------------
def test_three_indices_marked_bookkeeping_only():
    m = _load_mapping()
    ix = m["three_indices"]
    for key in ("i", "w", "k"):
        assert key in ix, f"three_indices missing {key}"
        assert ix[key].get("not_a_clock") is True, \
            f"three_indices.{key} must be declared not_a_clock=True"


def test_k_index_forbids_binding_to_either_architectural_clock():
    """The most important schema check for this tranche: the offline
    training index k MUST carry an explicit prohibition against
    being bound to either Aeon architectural clock."""
    m = _load_mapping()
    k = m["three_indices"]["k"]
    prohibited = set(k.get("must_not_map_to", []))
    assert "fast clock" in prohibited, \
        "three_indices.k must forbid mapping to 'fast clock'"
    assert "slow clock" in prohibited, \
        "three_indices.k must forbid mapping to 'slow clock'"


# ---------------------------------------------------------------------------
# 3. Prohibited mapping — slow clock != parameter update
# ---------------------------------------------------------------------------
def test_dual_clock_collision_prohibition_recorded():
    m = _load_mapping()
    p = m["dual_clock_collision_prohibition"]
    for field in ("prohibited_mapping", "why_prohibited",
                  "witnessed_absent_by_test", "spec_binding"):
        assert field in p, f"dual_clock_collision_prohibition missing {field}"
    assert "parameter update" in p["prohibited_mapping"].lower() or \
           "theta_{tau+1}" in p["prohibited_mapping"], \
        "prohibited_mapping must name the parameter-update equation"
    # The witness test must actually exist in the tree.
    witness_rel = p["witnessed_absent_by_test"]
    assert (ROOT / witness_rel).exists(), \
        f"witness test declared but missing: {witness_rel}"


def test_symbol_mapping_cross_references_closure_reference():
    sm = json.loads(SYMBOL_MAPPING_PATH.read_text(encoding="utf-8"))
    assert "closure_reference" in sm, \
        "symbol_mapping must contain a closure_reference block"
    cr = sm["closure_reference"]
    assert cr["role"] == "research_reference"
    assert cr["sha256"].startswith("sha256:")
    assert "prohibited_mapping" in cr
    for key in ("i", "w", "k"):
        assert key in cr["index_binding"]


# ---------------------------------------------------------------------------
# 4/5. Aeon fast + slow clock cadence — direct behavioral witness
# ---------------------------------------------------------------------------
def _build_small_hybrid_model():
    """Build the smallest possible HybridModel that still preserves
    the architectural invariants (K=16, sigma certificate). Uses a
    tiny transformer config so this test does not require the release
    bundle."""
    import torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    torch.manual_seed(20260822)
    tcfg = AeonTransformerConfig(
        vocab_size=32, hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=2, head_dim=16,
        intermediate_size=64, max_position_embeddings=128)
    return HybridModel(transformer_config=tcfg, h_rec=16, K=16,
                       margin_h=0.02, margin_c=0.02,
                       use_embedding_input=True, dtype=torch.float32)


def test_fast_substrate_step_called_exactly_once_per_token():
    """FAST CLOCK cadence — number of substrate.step calls must equal T."""
    import torch
    model = _build_small_hybrid_model()
    model.eval()

    call_count = {"n": 0}
    original_step = model.substrate.step
    def counting_step(x):
        call_count["n"] += 1
        return original_step(x)
    model.substrate.step = counting_step

    B, T = 1, 40  # not a multiple of K=16
    ids = torch.randint(low=0, high=32, size=(B, T), dtype=torch.long)
    with torch.inference_mode():
        _ = model(input_ids=ids)

    assert call_count["n"] == B * T, (
        f"FAST CLOCK cadence broken: substrate.step called "
        f"{call_count['n']} times; expected B*T = {B*T}")


def test_slow_recursion_step_called_exactly_once_per_K_window():
    """SLOW CLOCK cadence — RecursionJoiner.step call count must equal
    ceil(T / K). This is an independent second witness alongside
    tests/test_recursion_topology.py::test_recursion_step_called_once_per_window."""
    import torch
    model = _build_small_hybrid_model()
    model.eval()

    call_count = {"n": 0}
    original_step = model.recursion.step
    def counting_step(*a, **k):
        call_count["n"] += 1
        return original_step(*a, **k)
    model.recursion.step = counting_step

    B, T = 1, 40
    expected = math.ceil(T / 16)
    ids = torch.randint(low=0, high=32, size=(B, T), dtype=torch.long)
    with torch.inference_mode():
        _ = model(input_ids=ids)

    assert call_count["n"] == expected, (
        f"SLOW CLOCK cadence broken: recursion.step called "
        f"{call_count['n']} times; expected ceil(T/K) = {expected}")


# ---------------------------------------------------------------------------
# 6. Renderer changes cannot be counted as learned English
# ---------------------------------------------------------------------------
def test_renderer_path_does_not_touch_state_dict_or_parameters():
    """AST scan: the desktop runtime GENERATION + RENDERING path
    (the `_generate` method) must not contain any call to
    model.state_dict(...), model.parameters(...), model.load_state_dict(...),
    optimizer.step, .backward, .zero_grad, .requires_grad_, or any
    attribute assignment into ._parameters / ._buffers.

    Loading-time references inside `load_release` / `preflight` are
    fine — those are architecture setup, not rendering. Rendering
    manipulates strings (canonical decode, delta emission, U+FFFD
    hold-back) and MUST NOT be classifiable as parameter-learning.
    """
    src = RUNTIME_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(src)

    forbidden_call_attrs = {
        "state_dict", "load_state_dict", "parameters", "named_parameters",
        "requires_grad_", "zero_grad", "backward", "step",
    }

    # Find the `_generate` FunctionDef inside AeonDesktopRuntime.
    generate_node = None
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef) and cls.name == "AeonDesktopRuntime":
            for item in cls.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_generate":
                    generate_node = item
                    break
    assert generate_node is not None, \
        "Expected AeonDesktopRuntime._generate to exist"

    problems = []
    for node in ast.walk(generate_node):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in forbidden_call_attrs:
                # Allow rng.manual_seed()/torch.Generator().manual_seed()
                # and other non-model callables — those attrs are not in
                # the forbidden set anyway. If a forbidden name appears,
                # it is a real problem.
                problems.append(f".{f.attr}() at line {node.lineno}")
        if isinstance(node, ast.Attribute):
            if node.attr in ("_parameters", "_buffers"):
                problems.append(f".{node.attr} access at line {node.lineno}")
    assert not problems, (
        "AeonDesktopRuntime._generate (renderer + generation path) must "
        "not touch state_dict / parameters / grads / optimizer steps. "
        "Problems:\n  " + "\n  ".join(problems))


# ---------------------------------------------------------------------------
# 7. Protected P2 checkpoint hash unchanged
# ---------------------------------------------------------------------------
def test_protected_p2_checkpoint_hash_matches_freeze():
    fp = json.loads(ARCH_FREEZE_PATH.read_text(encoding="utf-8"))
    pinned = fp["protected_p2_checkpoint"]["sha256"]
    p2 = ROOT / "runs" / "aeon_lbc1_P2" / "final.pt"
    if not p2.exists():
        # State B — P2 not in this checkout. The pin is still recorded.
        return
    h = hashlib.sha256(); h.update(p2.read_bytes())
    assert f"sha256:{h.hexdigest()}" == pinned, (
        f"P2 checkpoint hash drifted:\n  pinned: {pinned}\n  disk : sha256:{h.hexdigest()}")


# ---------------------------------------------------------------------------
# 8. Tokenizer hash unchanged
# ---------------------------------------------------------------------------
def test_protected_tokenizer_hash_matches_freeze():
    fp = json.loads(ARCH_FREEZE_PATH.read_text(encoding="utf-8"))
    pinned = fp["protected_tokenizer"]["sha256"]
    tok = ROOT / "release-assets" / "aeon-desktop-p2-proxy" / "tokenizer" / "aeon-lbc1.model"
    if not tok.exists():
        return
    h = hashlib.sha256(); h.update(tok.read_bytes())
    assert f"sha256:{h.hexdigest()}" == pinned, (
        f"Tokenizer hash drifted:\n  pinned: {pinned}\n  disk : sha256:{h.hexdigest()}")


# ---------------------------------------------------------------------------
# 9. K remains exactly 16 in every consumer
# ---------------------------------------------------------------------------
def test_K_is_16_across_every_consumer():
    import yaml
    for p in ("configs/latent_bypass/aeon_lbc1_proxy.yaml",
              "configs/aeon_v1.yaml",
              "configs/aeon_350m.yaml"):
        cfg = yaml.safe_load((ROOT / p).read_text(encoding="utf-8"))
        K = cfg.get("model", {}).get("K", cfg.get("K"))
        assert K == 16, f"{p}: K = {K}, expected 16"
    # code default
    from aeon.hybrid import HybridModel
    import inspect
    sig = inspect.signature(HybridModel.__init__)
    assert sig.parameters["K"].default == 16
    # shuttle FIXED_K
    from aeon.shuttle import FIXED_K
    assert FIXED_K == 16


# ---------------------------------------------------------------------------
# 10. PDF SHA-256 declared in the mapping is a well-formed pin
# ---------------------------------------------------------------------------
def test_pdf_sha256_pin_is_wellformed():
    m = _load_mapping()
    sha = m["reference"]["sha256"]
    assert sha.startswith("sha256:"), "must be prefixed 'sha256:'"
    hexpart = sha.split(":", 1)[1]
    assert len(hexpart) == 64, f"expected 64 hex chars, got {len(hexpart)}"
    int(hexpart, 16)  # valid hex

    # Also present in the symbol_mapping cross-ref
    sm = json.loads(SYMBOL_MAPPING_PATH.read_text(encoding="utf-8"))
    assert sm["closure_reference"]["sha256"] == sha, \
        "closure_reference.sha256 must match between mapping and symbol_mapping"
