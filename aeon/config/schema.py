"""aeon/config/schema.py — user-config schema + validated atomic writes.

Fields (all optional at load; wizard fills them):
    tokenizer_path        : str absolute path
    corpus_path           : str absolute path
    checkpoint_dir        : str absolute path
    metrics_dir           : str absolute path
    evidence_dir          : str absolute path
    disk_allocation_gb    : int
    cpu_thread_limit      : int
    memory_ceiling_gb     : int (optional)
    checkpoint_interval   : int
    validation_interval   : int
    resume_preference     : "auto" | "never" | "prompt"
    training_config_id    : str (which certified training config to use)

Rules (§F4 + directive §W4):
    * No secrets, no arbitrary executable fields, no shell command fields.
    * No environment-variable expansion from untrusted values.
    * No path traversal — paths must be absolute.
    * Atomic writes.
    * Versioning + migration.
    * Canonical (sort_keys) JSON serialisation.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

USER_CONFIG_SCHEMA_VERSION = 1

_ALLOWED_RESUME = {"auto", "never", "prompt"}
_ALLOWED_TRAINING_CONFIGS = {
    "aeon_350m_primary.yaml", "aeon_smoke_e5.yaml", "aeon_v1.yaml",
}
# Whole-token / whole-word credential and code-execution field names. Split by
# non-alphanumeric characters so `tokenizer_path` (containing "tokenizer" but
# not the credential term "token") is not flagged. `access_token`, `api_key`,
# `signing_key`, etc. ARE flagged.
_FORBIDDEN_TOKENS = {
    "shell", "cmd", "command", "exec", "eval",
    "credential", "credentials",
    "password", "passwd", "passphrase",
    "secret", "secretkey", "apikey", "api_key",
    "bearer", "oauth", "authtoken", "auth_token",
    "signingkey", "signing_key", "certificate", "cert_password",
    "privatekey", "private_key",
}
import re as _re
_WORD_RE = _re.compile(r"[A-Za-z0-9]+")


def _is_forbidden_key(k: str) -> bool:
    """A key is forbidden if any of its underscore/dash-separated tokens exactly
    match a forbidden identifier. `tokenizer_path` → {"tokenizer", "path"} —
    neither matches; `access_token` → {"access", "token"} — "token" alone is not
    on the whole-token list; `auth_token` → matches "auth_token" via the full
    key normalization below."""
    lk = k.lower()
    tokens = set(_WORD_RE.findall(lk))
    if tokens & _FORBIDDEN_TOKENS:
        return True
    return lk in _FORBIDDEN_TOKENS


def _requires_absolute(path: str, field: str) -> Optional[str]:
    if not isinstance(path, str) or not path:
        return f"{field}: must be a non-empty string"
    # F4 refuses path traversal — no ".." components
    if any(part in ("..",) for part in re.split(r"[\\/]+", path)):
        return f"{field}: path traversal not allowed"
    if not os.path.isabs(path):
        return f"{field}: must be absolute (got {path!r})"
    return None


def validate_config_dict(cfg: Dict[str, Any]) -> List[str]:
    errs: List[str] = []
    if not isinstance(cfg, dict):
        return ["config is not a dict"]
    for k in cfg.keys():
        if _is_forbidden_key(k):
            errs.append(f"forbidden field: {k}")
    ver = cfg.get("schema_version")
    if ver is None:
        errs.append("missing schema_version")
    elif ver != USER_CONFIG_SCHEMA_VERSION:
        errs.append(f"unsupported schema_version {ver} (this Aeon knows {USER_CONFIG_SCHEMA_VERSION}); "
                    "use migrate_user_config()")
    # Path fields — validate when present; wizard fills at first-run
    for pf in ("tokenizer_path", "corpus_path", "checkpoint_dir",
                "metrics_dir", "evidence_dir"):
        v = cfg.get(pf)
        if v is None:
            continue
        e = _requires_absolute(v, pf)
        if e:
            errs.append(e)
    # Enum + numeric fields
    if "resume_preference" in cfg and cfg["resume_preference"] not in _ALLOWED_RESUME:
        errs.append(f"resume_preference: {cfg['resume_preference']!r} not in {sorted(_ALLOWED_RESUME)}")
    if "training_config_id" in cfg and cfg["training_config_id"] not in _ALLOWED_TRAINING_CONFIGS:
        errs.append(f"training_config_id: {cfg['training_config_id']!r} not in {sorted(_ALLOWED_TRAINING_CONFIGS)}")
    for nf in ("disk_allocation_gb", "cpu_thread_limit", "checkpoint_interval",
                "validation_interval", "memory_ceiling_gb"):
        v = cfg.get(nf)
        if v is None: continue
        if not isinstance(v, int) or v <= 0:
            errs.append(f"{nf}: must be a positive int (got {v!r})")
    return errs


def validate_config_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return [f"file not found: {path}"]
    try:
        cfg = json.load(open(path, encoding="utf-8"))
    except Exception as e:
        return [f"could not parse JSON: {e}"]
    return validate_config_dict(cfg)


def load_user_config(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def atomic_write_user_config(path: str, cfg: Dict[str, Any]) -> None:
    """Refuses invalid configs; writes atomically via temp+rename."""
    cfg = dict(cfg)
    cfg["schema_version"] = USER_CONFIG_SCHEMA_VERSION
    errs = validate_config_dict(cfg)
    if errs:
        raise ValueError("config invalid: " + "; ".join(errs))
    tmp = path + ".tmp"
    Path(os.path.dirname(path) or ".").mkdir(parents=True, exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, sort_keys=True, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def migrate_user_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Bring an older config forward to the current schema version. When the
    schema evolves, add per-version transformers here."""
    v = cfg.get("schema_version")
    if v == USER_CONFIG_SCHEMA_VERSION:
        return cfg
    out = dict(cfg)
    out["schema_version"] = USER_CONFIG_SCHEMA_VERSION
    return out
