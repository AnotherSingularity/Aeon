"""
F1 — Threat-model / asset-registry / boundary-registry schema validation.

Rejects incomplete threat entries per §F1.4. Torch-free.
"""
import copy
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_asset_registry_valid():
    from aeon.policies import load_asset_registry, validate_asset_registry
    errs = validate_asset_registry(load_asset_registry())
    assert not errs, "asset registry errors:\n  " + "\n  ".join(errs)


def test_threat_model_valid():
    from aeon.policies import load_threat_model, validate_threat_model
    errs = validate_threat_model(load_threat_model())
    assert not errs, "threat model errors:\n  " + "\n  ".join(errs)


def test_boundary_registry_valid():
    from aeon.policies import load_boundary_registry, validate_boundary_registry
    errs = validate_boundary_registry(load_boundary_registry())
    assert not errs, "boundary registry errors:\n  " + "\n  ".join(errs)


def test_schema_rejects_missing_threat_fields():
    """Confirm the validator ACTUALLY catches drift (not decorative)."""
    from aeon.policies import validate_threat_model, load_threat_model
    tm = copy.deepcopy(load_threat_model())
    del tm["adversaries"][0]["non_guarantees"]
    errs = validate_threat_model(tm)
    assert any("non_guarantees" in e for e in errs), errs
    tm2 = copy.deepcopy(load_threat_model())
    tm2["adversaries"] = tm2["adversaries"][:5]         # drop required adversaries
    errs2 = validate_threat_model(tm2)
    assert any("missing required adversaries" in e for e in errs2), errs2


def test_schema_rejects_missing_asset_fields():
    from aeon.policies import validate_asset_registry, load_asset_registry
    ar = copy.deepcopy(load_asset_registry())
    del ar["assets"][0]["confidentiality"]
    errs = validate_asset_registry(ar)
    assert any("confidentiality" in e for e in errs), errs


def test_schema_rejects_missing_boundary_fields():
    from aeon.policies import validate_boundary_registry, load_boundary_registry
    br = copy.deepcopy(load_boundary_registry())
    del br["boundaries"][0]["failure_behavior"]
    errs = validate_boundary_registry(br)
    assert any("failure_behavior" in e for e in errs), errs


def test_every_boundary_has_influence_flags():
    """Directive requires each boundary to declare whether data may influence
    model state and operational authority (both must be present as booleans)."""
    from aeon.policies import load_boundary_registry
    for b in load_boundary_registry()["boundaries"]:
        assert isinstance(b.get("may_influence_model_state"), bool), b["id"]
        assert isinstance(b.get("may_influence_operational_authority"), bool), b["id"]


def test_global_non_guarantees_present():
    from aeon.policies import load_threat_model
    tm = load_threat_model()
    gng = tm.get("non_guarantees_global", [])
    # Directive: MUST include the specific non-guarantees
    joined = " ".join(gng).lower()
    for token in ("processor", "firmware", "physical-memory", "hardware"):
        assert token in joined, f"missing non-guarantee about: {token}"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
