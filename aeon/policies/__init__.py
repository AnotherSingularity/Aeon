"""
aeon.policies — machine-readable policy loaders and schema validators.

Loads the F1 asset / threat / boundary registries (and later F3 checkpoint,
F4 runtime/security, F5 continuity policies) and validates their schemas.
Framework-free: this package imports no torch and does no I/O beyond json/os.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List

DOCS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs"))


class PolicyError(RuntimeError):
    """Raised when a policy registry is missing a required field or is malformed."""


def _load(name: str) -> Dict[str, Any]:
    path = os.path.join(DOCS_ROOT, name)
    if not os.path.exists(path):
        raise PolicyError(f"missing policy file: {path}")
    return json.load(open(path))


def load_asset_registry() -> Dict[str, Any]:
    return _load("asset_registry.json")


def load_threat_model() -> Dict[str, Any]:
    return _load("threat_model.json")


def load_boundary_registry() -> Dict[str, Any]:
    return _load("boundary_registry.json")


REQUIRED_ASSET_FIELDS = ["id", "name", "kind", "location", "confidentiality",
                          "integrity", "availability", "authenticity",
                          "provenance", "recovery", "rollback", "retention_days",
                          "authorized_disclosure"]

REQUIRED_THREAT_FIELDS = ["id", "name", "access", "knowledge", "assets_at_risk",
                          "expected_response", "residual_risk", "non_guarantees"]

REQUIRED_BOUNDARY_FIELDS = ["id", "name", "permitted_input", "required_identity",
                            "validation", "size_limits_mib", "failure_behavior",
                            "audit_required", "may_influence_model_state",
                            "may_influence_operational_authority"]

REQUIRED_ASSET_KINDS = {"source_code", "build_scripts", "dependency_lockfile",
                        "architecture_config", "model_weights", "recursion_state",
                        "substrate_state", "optimizer_state", "scheduler_state",
                        "rng_state", "training_corpus", "corpus_manifest",
                        "tokenizer", "evaluation_data", "training_config",
                        "runtime_policy", "checkpoint_envelope", "metrics",
                        "audit_records", "cryptographic_keys", "credentials",
                        "recovery_artifact", "manufacturing_telemetry",
                        "communications_telemetry", "operator_decisions",
                        "update_packages"}

REQUIRED_ADVERSARY_NAMES = {
    "accidental_operator_error", "malformed_input", "poisoned_corpus_contribution",
    "unauthorized_local_user", "compromised_dependency", "compromised_update_package",
    "remote_attacker", "privileged_host_compromise", "powered_off_device_theft",
    "checkpoint_substitution", "checkpoint_rollback", "model_extraction_attempt",
    "resource_exhaustion", "audit_tampering", "unauthorized_config_change",
    "recovery_state_corruption", "replay_of_previously_valid_state",
    "incompatible_valid_artifact_injection"}

REQUIRED_BOUNDARY_NAMES = {
    "corpus_ingestion", "tokenizer", "training_process", "model_runtime",
    "recursion_state", "checkpoint", "key_management", "operator", "evaluation",
    "network", "filesystem", "update", "recovery", "manufacturing_interface",
    "communications_interface", "audit"}


def validate_asset_registry(reg: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    assets = reg.get("assets", [])
    seen_ids: set = set()
    for a in assets:
        for f in REQUIRED_ASSET_FIELDS:
            if f not in a:
                errors.append(f"asset {a.get('id', '?')}: missing field {f}")
        if a.get("id") in seen_ids:
            errors.append(f"duplicate asset id: {a['id']}")
        seen_ids.add(a.get("id"))
    covered_names = {a.get("name") for a in assets}
    missing = REQUIRED_ASSET_KINDS - covered_names
    if missing:
        # allow synonyms: source_code / build_scripts etc. are exact-name entries
        errors.append(f"asset registry missing required names: {sorted(missing)}")
    return errors


def validate_threat_model(tm: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    advs = tm.get("adversaries", [])
    seen_ids: set = set()
    for t in advs:
        for f in REQUIRED_THREAT_FIELDS:
            if f not in t:
                errors.append(f"threat {t.get('id', '?')}: missing field {f}")
        if not isinstance(t.get("non_guarantees"), list) or not t["non_guarantees"]:
            errors.append(f"threat {t.get('id', '?')}: non_guarantees must be a non-empty list")
        if not isinstance(t.get("assets_at_risk"), list) or not t["assets_at_risk"]:
            errors.append(f"threat {t.get('id', '?')}: assets_at_risk must be non-empty")
        if t.get("id") in seen_ids:
            errors.append(f"duplicate threat id: {t['id']}")
        seen_ids.add(t.get("id"))
    covered = {t.get("name") for t in advs}
    missing = REQUIRED_ADVERSARY_NAMES - covered
    if missing:
        errors.append(f"threat model missing required adversaries: {sorted(missing)}")
    if not tm.get("non_guarantees_global"):
        errors.append("missing non_guarantees_global")
    return errors


def validate_boundary_registry(br: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    seen_ids: set = set()
    for b in br.get("boundaries", []):
        for f in REQUIRED_BOUNDARY_FIELDS:
            if f not in b:
                errors.append(f"boundary {b.get('id', '?')}: missing field {f}")
        if b.get("id") in seen_ids:
            errors.append(f"duplicate boundary id: {b['id']}")
        seen_ids.add(b.get("id"))
    covered = {b.get("name") for b in br.get("boundaries", [])}
    missing = REQUIRED_BOUNDARY_NAMES - covered
    if missing:
        errors.append(f"boundary registry missing required boundaries: {sorted(missing)}")
    return errors


def validate_all() -> List[str]:
    """Return a list of errors across every registry. Empty list == valid."""
    errs: List[str] = []
    errs += validate_asset_registry(load_asset_registry())
    errs += validate_threat_model(load_threat_model())
    errs += validate_boundary_registry(load_boundary_registry())
    return errs
