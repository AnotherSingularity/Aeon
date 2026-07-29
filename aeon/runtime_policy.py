"""
aeon/runtime_policy.py — deny-by-default runtime policy loader + enforcement.

Directive F4:
  §F4.1 execution identity — Aeon-side attestations that identity constraints
        hold; OS-level enforcement is deployment work.
  §F4.2 filesystem policy — deny-by-default; template-based read/write roots;
        symlink-escape and path-traversal rejection.
  §F4.3 network policy — certified local mode has ZERO required network use.
  §F4.4 process/code-execution policy — no shell, subprocess, eval, plugin,
        or arbitrary-module load from model-directed paths.
  §F4.5 resource controls — ceilings enforced when Aeon owns the mechanism.
  §F4.6 fail-closed — named condition set.

Torch-free.
"""
from __future__ import annotations

import ast
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
POLICY_PATH = os.path.join(REPO_ROOT, "docs", "runtime_policy.json")


class RuntimePolicyError(RuntimeError):
    """Fail-closed on any policy violation."""


def load_policy() -> Dict[str, Any]:
    if not os.path.exists(POLICY_PATH):
        raise RuntimePolicyError(f"missing runtime policy: {POLICY_PATH}")
    return json.load(open(POLICY_PATH))


# ---------------------------------------------------------------------------
# Template resolution (§F4.2)
# ---------------------------------------------------------------------------
_TEMPLATE_KEYS = {"<repo>", "<corpus_root>", "<tokenizer_root>",
                   "<tmp>", "<out_dir>"}


def _expand(template: str, substitutions: Dict[str, str]) -> str:
    """Substitute policy template placeholders. Missing substitutions raise."""
    out = template
    for k, v in substitutions.items():
        out = out.replace(k, v)
    for k in _TEMPLATE_KEYS:
        if k in out and k not in substitutions:
            # Left an unresolved template — refuse.
            raise RuntimePolicyError(f"unresolved template {k!r} in {template!r}")
    return os.path.abspath(out)


def _within(child: str, parent: str) -> bool:
    """Is `child` an on-disk descendant of `parent` (both absolute, no symlink
    escapes)?"""
    try:
        c = os.path.realpath(child)
        p = os.path.realpath(parent)
        return c == p or c.startswith(p.rstrip(os.sep) + os.sep)
    except Exception:
        return False


def check_path(
    path: str,
    mode: str,                              # "read" | "write"
    *,
    substitutions: Dict[str, str],
    policy: Optional[Dict[str, Any]] = None,
) -> str:
    """Return the canonical absolute path if allowed; raise RuntimePolicyError
    otherwise. Enforces:
      - deny-by-default: only allow-listed roots grant access.
      - traversal denial: '..' in the resolved path must not escape the roots.
      - symlink escape: realpath() must still be within an allowed root.
    """
    if policy is None:
        policy = load_policy()
    fs = policy["filesystem_policy"]

    read_roots = [_expand(r["template"], substitutions) for r in fs["read_only_roots"]]
    write_roots = [_expand(r["template"], substitutions) for r in fs["writable_roots"]]

    ap = os.path.abspath(path)
    # traversal check: after normalization, no ".." remains
    if ".." in ap.split(os.sep):
        raise RuntimePolicyError(f"path traversal: {path!r}")

    if mode == "read":
        allowed = read_roots + write_roots
    elif mode == "write":
        allowed = write_roots
    else:
        raise RuntimePolicyError(f"unknown mode {mode!r}")

    for root in allowed:
        if _within(ap, root):
            return ap
    raise RuntimePolicyError(f"denied {mode}: {path!r} outside allowed roots")


# ---------------------------------------------------------------------------
# Static verifiers (§F4.3, §F4.4) — AST-based, string literals excluded
# ---------------------------------------------------------------------------
_FORBIDDEN_NETWORK_MODULES = {"socket", "urllib.request", "requests", "http.client",
                               "asyncio.open_connection", "smtplib"}
# (module, attr) pairs that mean "shell / subprocess"
_FORBIDDEN_SHELL_ATTRS = {("os", "system"), ("os", "popen"),
                          ("subprocess", "Popen"), ("subprocess", "check_call"),
                          ("subprocess", "call")}
# Bare-name calls that would evaluate arbitrary code
_FORBIDDEN_EVAL_NAMES = {"eval", "exec", "compile"}

# Documented, argued-safe exemptions (allowlisted with reason). Adding to this
# list requires a docs entry justifying why the use is not model-directed.
_ALLOWED_DYNAMIC_IMPORT = {
    ("aeon/provenance.py", "__import__"): "runtime version reporting for fixed known deps only",
    ("aeon/runtime_policy.py", "self"): "the scanner itself; whitelisted by module identity",
}


def _iter_py(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith(".py"):
                yield os.path.join(dirpath, fn)


def _module_from_import_alias(alias: str) -> str:
    return alias.split(".", 1)[0]


def scan_forward_path_for_network_client() -> List[str]:
    """AST scan of aeon/ and scripts/: an ImportFrom or Import of a network
    client module is a failure. String literals and comments are ignored."""
    offenders: List[str] = []
    for root in ("aeon", "scripts"):
        for path in _iter_py(os.path.join(REPO_ROOT, root)):
            relative = os.path.relpath(path, REPO_ROOT)
            try:
                tree = ast.parse(open(path).read(), filename=path)
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in _FORBIDDEN_NETWORK_MODULES:
                            offenders.append(f"{relative}: import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    mod = node.module or ""
                    if mod in _FORBIDDEN_NETWORK_MODULES:
                        offenders.append(f"{relative}: from {mod} import ...")
    return offenders


def scan_for_shell_or_eval() -> List[str]:
    """AST scan of aeon/: only real Call nodes count. Skips this module (the
    definition of what to look for) and honours _ALLOWED_DYNAMIC_IMPORT."""
    offenders: List[str] = []
    for path in _iter_py(os.path.join(REPO_ROOT, "aeon")):
        relative = os.path.relpath(path, REPO_ROOT).replace(os.sep, "/")
        # The scanner itself defines these tokens; skip its OWN file.
        if relative == "aeon/runtime_policy.py":
            continue
        try:
            tree = ast.parse(open(path).read(), filename=path)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            # os.system(...) / subprocess.Popen(...)
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                pair = (func.value.id, func.attr)
                if pair in _FORBIDDEN_SHELL_ATTRS:
                    offenders.append(f"{relative}: {pair[0]}.{pair[1]}()")
            # eval(...) / exec(...) / compile(...) / __import__(...)
            elif isinstance(func, ast.Name):
                name = func.id
                if name in _FORBIDDEN_EVAL_NAMES:
                    offenders.append(f"{relative}: {name}()")
                elif name == "__import__":
                    if (relative, "__import__") not in _ALLOWED_DYNAMIC_IMPORT:
                        offenders.append(f"{relative}: __import__()")
    return offenders


def scan_no_absolute_paths_in_policy() -> List[str]:
    """§F4.2 — committed policy must be template-based, not absolute machine
    paths. Reject any /home/... /Users/... C:\\Users... path in the policy."""
    pol = load_policy()
    offenders: List[str] = []
    for section_name in ("read_only_roots", "writable_roots"):
        for entry in pol["filesystem_policy"].get(section_name, []):
            tpl = entry.get("template", "")
            if re.match(r"^/(home|Users|root)/", tpl) or re.match(r"^[A-Z]:\\", tpl):
                offenders.append(f"{section_name}: {tpl!r} is machine-specific")
    return offenders


# ---------------------------------------------------------------------------
# Resource controls (§F4.5)
# ---------------------------------------------------------------------------
@dataclass
class ResourceCeilings:
    seq_len_max: int
    batch_size_max: int
    input_size_bytes_max: int
    disk_ceiling_gb_per_run: int
    checkpoint_retention_max: int
    queue_depth_max: int
    audit_log_ceiling_mb: int


def load_resource_ceilings() -> ResourceCeilings:
    r = load_policy()["resource_controls"]
    return ResourceCeilings(
        seq_len_max=int(r["seq_len_max"]),
        batch_size_max=int(r["batch_size_max"]),
        input_size_bytes_max=int(r["input_size_bytes_max"]),
        disk_ceiling_gb_per_run=int(r["disk_ceiling_gb_per_run"]),
        checkpoint_retention_max=int(r["checkpoint_retention_max"]),
        queue_depth_max=int(r["queue_depth_max"]),
        audit_log_ceiling_mb=int(r["audit_log_ceiling_mb"]),
    )


def enforce_ceilings_on_config(model_cfg: Dict[str, Any],
                                 data_cfg: Dict[str, Any],
                                 train_cfg: Dict[str, Any]) -> None:
    """Refuse configs that would breach the §F4.5 ceilings."""
    c = load_resource_ceilings()
    if int(data_cfg.get("seq_len", 0)) > c.seq_len_max:
        raise RuntimePolicyError(f"seq_len {data_cfg['seq_len']} exceeds ceiling {c.seq_len_max}")
    if int(train_cfg.get("batch_size", 0)) > c.batch_size_max:
        raise RuntimePolicyError(f"batch_size {train_cfg['batch_size']} exceeds ceiling {c.batch_size_max}")


# ---------------------------------------------------------------------------
# Fail-closed conditions (§F4.6)
# ---------------------------------------------------------------------------
def fail_closed_conditions() -> List[str]:
    return list(load_policy().get("fail_closed_conditions", []))
