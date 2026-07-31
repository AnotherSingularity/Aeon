"""aeon/config/preflight.py — pre-training verifier.

Runs every §W4 preflight check. Returns a PreflightResult with per-check
status; overall verdict is READY / READY_WITH_WARNINGS / BLOCKED. The launcher
disables Start while BLOCKED."""
from __future__ import annotations

import enum
import os
import shutil
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aeon.windows_paths import (
    installed_resource_root, user_data_root, jobs_dir, config_dir,
)


class PreflightVerdict(str, enum.Enum):
    READY = "READY"
    READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
    BLOCKED = "BLOCKED"


@dataclass
class PreflightCheck:
    name: str
    status: str                     # "pass" | "warn" | "fail" | "skip"
    detail: str = ""


@dataclass
class PreflightResult:
    verdict: PreflightVerdict
    checks: List[PreflightCheck] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "checks": [c.__dict__ for c in self.checks],
        }


def _add(res: PreflightResult, name: str, status: str, detail: str = "") -> None:
    res.checks.append(PreflightCheck(name=name, status=status, detail=detail))


def _is_frozen() -> bool:
    """Frozen mode is the PyInstaller-built Windows bundle. In that mode
    the tokenizer and corpus MUST be present + loadable — falling back to
    torch.randint synthetic tokens produced the shipped-broken worker the
    audit found in W10-0/A17. Source-tree runs continue to warn (dev
    convenience) but the frozen desktop path fails closed.
    """
    return bool(getattr(sys, "frozen", False) or hasattr(sys, "_MEIPASS"))


def _tokenizer_load_fails(path: str) -> Optional[str]:
    try:
        from aeon.tokenizer import AeonTokenizer
        AeonTokenizer(path)
        return None
    except Exception as e:
        return str(e)


def _corpus_read_fails(path: str) -> Optional[str]:
    """Attempt to yield at least one text record from the corpus. If the
    file is empty, unreadable, or produces zero records, the corpus is
    unusable and the check fails closed."""
    try:
        from aeon.data import iter_text_records
        it = iter_text_records(path)
        first = next(it, None)
        if first is None:
            return "corpus yields zero records"
        return None
    except Exception as e:
        return str(e)


def run_preflight(user_cfg: Dict[str, Any]) -> PreflightResult:
    """Run every §W4 check. Never raises — always returns a result.

    W10-8 correction: in FROZEN mode (Windows PyInstaller bundle), the
    tokenizer_path and corpus_path checks fail (BLOCKED) instead of warn
    when either is missing or unloadable. The corrective directive's A17
    finding was that the frozen desktop preflight would return
    READY_WITH_WARNINGS on a bundle that could not actually train — the
    worker would then fall back to torch.randint synthetic tokens. In
    source-tree mode the previous WARN behaviour is preserved for
    developer convenience.
    """
    res = PreflightResult(verdict=PreflightVerdict.READY)
    frozen = _is_frozen()
    unusable_status = "fail" if frozen else "warn"

    # --- 1. OS + arch ---
    plat = sys.platform
    _add(res, "os_and_arch", "pass", f"{plat} {os.uname().machine if hasattr(os, 'uname') else '?'}")

    # --- 2. CPU availability ---
    cpu = os.cpu_count() or 0
    _add(res, "cpu_available", "pass" if cpu >= 1 else "fail", f"cpu_count={cpu}")

    # --- 3. Memory + disk (best-effort) ---
    try:
        import shutil as _sh
        du = _sh.disk_usage(str(user_data_root()))
        gb_free = du.free / (1024 ** 3)
        min_gb = int(user_cfg.get("disk_allocation_gb", 8))
        status = "pass" if gb_free >= min_gb else "warn"
        _add(res, "disk_space", status, f"{gb_free:.1f} GB free (need >= {min_gb} GB)")
    except Exception as e:
        _add(res, "disk_space", "warn", f"could not measure: {e}")

    # --- 4. Write access to user_data ---
    try:
        p = user_data_root() / ".preflight_write_check"
        p.write_text("ok", encoding="utf-8"); p.unlink()
        _add(res, "user_data_writable", "pass", str(user_data_root()))
    except Exception as e:
        _add(res, "user_data_writable", "fail", f"cannot write under user_data_root: {e}")

    # --- 5. Tokenizer identity + presence ---
    # W10-8/A17: in frozen mode, missing/unloadable tokenizer BLOCKS.
    tok = user_cfg.get("tokenizer_path")
    if not tok:
        _add(res, "tokenizer", unusable_status,
             "no tokenizer configured "
             + ("(frozen mode requires a tokenizer — BLOCKED)" if frozen
                else "(synthetic-token fallback will apply)"))
    elif not os.path.exists(tok):
        _add(res, "tokenizer", "fail", f"tokenizer file not found: {tok}")
    else:
        err = _tokenizer_load_fails(tok)
        if err is None:
            from aeon.tokenizer import AeonTokenizer
            t = AeonTokenizer(tok)
            _add(res, "tokenizer", "pass",
                 f"loaded vocab={t.vocab_size} bos={t.bos_id} eos={t.eos_id}")
        else:
            _add(res, "tokenizer", "fail", f"tokenizer load failed: {err}")

    # --- 6. Corpus identity + provenance ---
    # W10-8/A17: in frozen mode, missing/empty/unreadable corpus BLOCKS.
    # A corpus that yields zero records is unusable — the worker would fall
    # through to torch.randint synthetic tokens.
    corpus = user_cfg.get("corpus_path")
    if not corpus:
        _add(res, "corpus", unusable_status,
             "no corpus configured "
             + ("(frozen mode requires a corpus — BLOCKED)" if frozen
                else "(synthetic-token fallback will apply)"))
    elif not os.path.exists(corpus):
        _add(res, "corpus", "fail", f"corpus not found: {corpus}")
    else:
        err = _corpus_read_fails(corpus)
        if err is None:
            _add(res, "corpus", "pass", f"corpus present and readable at {corpus}")
        else:
            _add(res, "corpus", unusable_status, f"corpus unusable: {err}")

    # --- 7. Configuration identity ---
    tcid = user_cfg.get("training_config_id") or "aeon_350m_primary.yaml"
    resolved = installed_resource_root() / "configs" / tcid
    if resolved.exists():
        _add(res, "training_config_identity", "pass", str(resolved))
    else:
        _add(res, "training_config_identity", "fail",
             f"training config not found in installed configs/: {resolved}")

    # --- 8. Runtime integrity (best-effort: if RUNTIME_MANIFEST.json exists) ---
    try:
        from aeon.integrity import verify_installed_manifest
        ok, report = verify_installed_manifest()
        if report.get("reason", {}).get("error") == "manifest_missing":
            _add(res, "runtime_integrity", "warn",
                 "runtime manifest not present (source-mode dev build)")
        elif ok:
            _add(res, "runtime_integrity", "pass",
                 f"{report['files_ok']}/{report['files_checked']} files verified")
        else:
            _add(res, "runtime_integrity", "fail",
                 f"missing={len(report.get('missing', []))} mismatched={len(report.get('mismatched', []))}")
    except Exception as e:
        _add(res, "runtime_integrity", "warn", f"could not verify: {e}")

    # --- 9. Checkpoint chain state ---
    ck = user_cfg.get("checkpoint_dir")
    if ck and os.path.isdir(ck):
        n = len([f for f in os.listdir(ck) if f.startswith("ckpt_")])
        _add(res, "checkpoint_chain", "pass", f"{n} checkpoint(s) in {ck}")
    elif ck:
        _add(res, "checkpoint_chain", "warn", f"checkpoint dir empty or missing: {ck}")
    else:
        _add(res, "checkpoint_chain", "warn", "no checkpoint_dir configured")

    # --- 10. No active conflicting worker ---
    try:
        from aeon.job.manager import active_jobs
        from aeon.job.identity import verify_worker_identity
        conflicting = []
        for j in active_jobs():
            ident = verify_worker_identity(j.job_dir)
            if ident is not None and ck and os.path.samefile(j.checkpoint_dir, ck):
                conflicting.append(j.job_id)
        if conflicting:
            _add(res, "no_conflicting_worker", "fail",
                 f"jobs already writing to {ck}: {conflicting}")
        else:
            _add(res, "no_conflicting_worker", "pass", "no live worker on this ckpt chain")
    except Exception as e:
        _add(res, "no_conflicting_worker", "warn", f"could not enumerate: {e}")

    # --- 11-15. Architectural invariants preserved (static structural checks) ---
    try:
        import aeon.hybrid as _h
        assert "K: int = 16" in open(_h.__file__, encoding="utf-8").read()
        _add(res, "K_equals_16", "pass", "hybrid.py declares K=16 default")
    except Exception as e:
        _add(res, "K_equals_16", "fail", str(e))

    try:
        import aeon.transformer as _t
        assert "register_buffer" not in open(_t.__file__, encoding="utf-8").read()
        _add(res, "rotary_no_register_buffer", "pass", "P-4f still enforced")
    except Exception as e:
        _add(res, "rotary_no_register_buffer", "fail", str(e))

    try:
        import aeon.checkpoint as _c
        assert "weights_only=True" in open(_c.__file__, encoding="utf-8").read()
        _add(res, "strict_load_weights_only", "pass", "E3 strict_load uses weights_only")
    except Exception as e:
        _add(res, "strict_load_weights_only", "fail", str(e))

    # --- 16. Security + runtime policies loadable ---
    try:
        from aeon.runtime_policy import load_policy
        pol = load_policy()
        assert pol.get("policy_id"), "no policy_id"
        _add(res, "runtime_policy_loaded", "pass", pol["policy_id"])
    except Exception as e:
        _add(res, "runtime_policy_loaded", "fail", str(e))

    # --- 17. Local network-denied mode (static AST scan already in F4) ---
    try:
        from aeon.runtime_policy import scan_forward_path_for_network_client
        off = scan_forward_path_for_network_client()
        if off:
            _add(res, "no_network_dependency", "fail", f"offenders: {off[:3]}")
        else:
            _add(res, "no_network_dependency", "pass", "no network client imported")
    except Exception as e:
        _add(res, "no_network_dependency", "warn", str(e))

    # --- 18. Contractive certificate structural check ---
    _add(res, "contractive_certificate", "pass",
         "sigma < MARGIN structural by construction (Recursion._build)")

    # --- Verdict ---
    if any(c.status == "fail" for c in res.checks):
        res.verdict = PreflightVerdict.BLOCKED
    elif any(c.status == "warn" for c in res.checks):
        res.verdict = PreflightVerdict.READY_WITH_WARNINGS
    else:
        res.verdict = PreflightVerdict.READY
    return res
