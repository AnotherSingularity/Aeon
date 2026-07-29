"""
aeon/provenance.py — canonical artefact identity and provenance chain.

Directive F2 requires:
  * Stable, portable identities for every artefact.
  * Canonical serialization before hashing (§F2.1 — identity must not shift
    with absolute paths, hostnames, tempdirs, non-semantic key ordering, or
    incidental timestamp formatting).
  * A provenance chain: Source → Build → Config → Tokenizer → Corpus →
    TrainingRun → Checkpoint → Evaluation → Recovery. Every downstream
    artefact refers to its immediate authenticated inputs (§F2.2).
  * Refusal on missing / mismatched provenance (§F2.5).

Framework-free. Uses only the Python stdlib (hashlib, json, os, subprocess).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Identity fields the F2 provenance chain considers SEMANTIC. Any other keys
# (paths, hostnames, wall-clock timestamps, temp dirs) are stripped by
# canonicalize() before hashing so identical semantic content produces
# identical identity regardless of environment.
_ENV_STRIP_KEYS = {"absolute_path", "hostname", "tmpdir", "cwd", "run_started_at",
                    "wall_clock", "epoch", "container_id", "user"}


class ProvenanceError(RuntimeError):
    """Raised when required provenance is missing, mismatched, or malformed."""


# ---------------------------------------------------------------------------
# Canonical serialization
# ---------------------------------------------------------------------------
def canonicalize(obj: Any) -> Any:
    """Return a semantically-equivalent structure with environment-specific
    incidentals removed.

    Rules:
      - dict: keys sorted; env-incidental keys stripped; values canonicalised.
      - list/tuple: elements canonicalised in order (order IS semantic).
      - float: kept as-is (identity is byte-level via JSON with sort_keys).
      - str/int/bool/None: kept as-is.
    """
    if isinstance(obj, dict):
        return {k: canonicalize(v) for k, v in sorted(obj.items())
                if k not in _ENV_STRIP_KEYS}
    if isinstance(obj, (list, tuple)):
        return [canonicalize(x) for x in obj]
    return obj


def _canonical_json(obj: Any) -> bytes:
    """Canonical JSON: sorted keys, no whitespace variance, UTF-8 bytes."""
    return json.dumps(canonicalize(obj), sort_keys=True,
                      separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def hash_object(obj: Any) -> str:
    """Semantic sha256 over canonical-JSON of an object."""
    return hashlib.sha256(_canonical_json(obj)).hexdigest()


def hash_file(path: str, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Identity constructors (one per §F2.1 kind)
# ---------------------------------------------------------------------------
def source_commit_identity() -> Dict[str, Any]:
    try:
        rev = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=2, cwd=REPO_ROOT).stdout.strip()
    except Exception:
        rev = ""
    try:
        dirty = bool(subprocess.run(["git", "status", "--porcelain"],
                                    capture_output=True, text=True, timeout=2,
                                    cwd=REPO_ROOT).stdout.strip())
    except Exception:
        dirty = True
    return {"kind": "source_commit", "commit": rev or "unknown", "dirty": bool(dirty)}


def dependency_lockfile_identity() -> Dict[str, Any]:
    py = os.path.join(REPO_ROOT, "pyproject.toml")
    return {"kind": "dependency_lockfile", "path": "pyproject.toml",
            "sha256": hash_file(py) if os.path.exists(py) else None}


def runtime_versions_identity() -> Dict[str, Any]:
    versions: Dict[str, str] = {}
    for name in ("torch", "sentencepiece", "yaml", "numpy"):
        try:
            m = __import__(name)
            versions[name] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[name] = "MISSING"
    try:
        import sys
        versions["python"] = sys.version.split()[0]
    except Exception:
        pass
    return {"kind": "runtime_versions", "versions": versions}


def build_configuration_identity() -> Dict[str, Any]:
    return {"kind": "build_configuration",
            "dependency_lockfile": dependency_lockfile_identity(),
            "runtime_versions": runtime_versions_identity()}


def config_identity(cfg: Dict[str, Any], kind: str = "model_configuration") -> Dict[str, Any]:
    return {"kind": kind, "config_sha256": hash_object(cfg)}


def tokenizer_identity(path: Optional[str]) -> Dict[str, Any]:
    if not path or not os.path.exists(path):
        return {"kind": "tokenizer", "present": False, "sha256": None}
    ident: Dict[str, Any] = {"kind": "tokenizer", "present": True,
                              "sha256": hash_file(path)}
    try:
        from aeon.tokenizer import AeonTokenizer
        tok = AeonTokenizer(path)
        ident["vocab_size"] = int(tok.vocab_size)
        ident["special_ids"] = {"pad": tok.pad_id, "unk": tok.unk_id,
                                 "bos": tok.bos_id, "eos": tok.eos_id}
    except Exception as e:
        ident["load_error"] = str(e)
    return ident


def corpus_manifest_identity(manifest: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": "corpus_manifest", "manifest_sha256": hash_object(manifest),
            "n_sources": len(manifest.get("sources", []))}


def policy_identity(kind: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    return {"kind": kind, "policy_sha256": hash_object(obj)}


# ---------------------------------------------------------------------------
# Provenance chain: Source → Build → Config → Tokenizer → Corpus →
#                  TrainingRun → Checkpoint → Evaluation → Recovery
# ---------------------------------------------------------------------------
CHAIN_KINDS = ("source_commit", "build_configuration", "model_configuration",
               "tokenizer", "corpus_manifest", "training_run", "checkpoint",
               "evaluation", "recovery")


@dataclass
class ProvenanceRecord:
    """Complete provenance for a downstream artefact. Every field required per
    §F2.2 — any None field for a checkpoint or downstream artefact triggers
    refusal in strict_verify()."""
    source_commit: Dict[str, Any] = field(default_factory=dict)
    build_configuration: Dict[str, Any] = field(default_factory=dict)
    model_configuration: Dict[str, Any] = field(default_factory=dict)
    tokenizer: Dict[str, Any] = field(default_factory=dict)
    corpus_manifest: Dict[str, Any] = field(default_factory=dict)
    training_run: Optional[Dict[str, Any]] = None
    checkpoint: Optional[Dict[str, Any]] = None
    evaluation: Optional[Dict[str, Any]] = None
    recovery: Optional[Dict[str, Any]] = None
    runtime_policy: Dict[str, Any] = field(default_factory=dict)
    security_policy: Dict[str, Any] = field(default_factory=dict)
    patch_manifest_version: int = 1
    architecture_manifest_version: int = 1

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}

    def provenance_hash(self) -> str:
        """Semantic sha256 over the whole provenance record."""
        return hash_object(self.to_dict())


# ---------------------------------------------------------------------------
# Verification (§F2.5)
# ---------------------------------------------------------------------------
_REQUIRED_FOR_CHECKPOINT = ("source_commit", "build_configuration",
                             "model_configuration", "tokenizer",
                             "corpus_manifest", "training_run",
                             "runtime_policy", "security_policy")


def strict_verify(rec: Dict[str, Any], *, kind: str) -> None:
    """Refuse artefacts with missing / incoherent provenance.

    kind: what the downstream artefact IS ("checkpoint", "recovery", "eval").
    Raises ProvenanceError with a named reason. Never returns False silently.
    """
    if kind == "checkpoint":
        for k in _REQUIRED_FOR_CHECKPOINT:
            v = rec.get(k)
            if not v or (isinstance(v, dict) and not v):
                raise ProvenanceError(f"missing provenance field: {k}")
        # tokenizer present but no sha256 == not identifiable
        tok = rec.get("tokenizer", {})
        if tok.get("present") and not tok.get("sha256"):
            raise ProvenanceError("tokenizer marked present but no sha256")
        # source commit must be non-empty and not literal 'unknown'
        sc = rec.get("source_commit", {})
        if not sc.get("commit") or sc.get("commit") == "unknown":
            raise ProvenanceError("source commit not identifiable")
        if sc.get("dirty"):
            # dirty tree is allowed in dev but MUST be flagged in provenance
            # — the record itself carries the flag so a downstream check can
            # refuse dirty-provenance in production. We do not raise here.
            pass
    elif kind == "recovery":
        # A recovery record must at minimum reference the checkpoint it restores.
        if not rec.get("recovery"):
            raise ProvenanceError("recovery record missing 'recovery' section")
        if not rec["recovery"].get("selected_state_identity"):
            raise ProvenanceError("recovery missing selected_state_identity")
    elif kind == "eval":
        if not rec.get("evaluation"):
            raise ProvenanceError("evaluation record missing 'evaluation' section")
    else:
        raise ProvenanceError(f"unknown provenance kind: {kind}")


# ---------------------------------------------------------------------------
# Convenience: build a ProvenanceRecord for a training run in one call.
# ---------------------------------------------------------------------------
def build_training_provenance(
    *,
    model_cfg: Dict[str, Any],
    tokenizer_path: Optional[str],
    corpus_manifest: Dict[str, Any],
    training_run_info: Dict[str, Any],
    runtime_policy: Dict[str, Any],
    security_policy: Dict[str, Any],
    patch_manifest_version: int = 1,
    architecture_manifest_version: int = 1,
) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_commit=source_commit_identity(),
        build_configuration=build_configuration_identity(),
        model_configuration=config_identity(model_cfg, "model_configuration"),
        tokenizer=tokenizer_identity(tokenizer_path),
        corpus_manifest=corpus_manifest_identity(corpus_manifest),
        training_run=training_run_info,
        runtime_policy=policy_identity("runtime_policy", runtime_policy),
        security_policy=policy_identity("security_policy", security_policy),
        patch_manifest_version=patch_manifest_version,
        architecture_manifest_version=architecture_manifest_version,
    )
