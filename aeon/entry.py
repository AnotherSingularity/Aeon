"""
aeon/entry.py — one frozen entry point that dispatches to GUI or internal modes.

Installed executable behaviour (Aeon.exe):

    Aeon.exe                            → launch GUI (default)
    Aeon.exe --worker <job-file>        → run training worker (internal)
    Aeon.exe --verify-installation      → verify runtime manifest + exit
    Aeon.exe --validate-config <cfg>    → schema-check a config + exit
    Aeon.exe --diagnose <checkpoint>    → offline diagnostics (internal)
    Aeon.exe --recover <request>        → operator-authorised recovery (internal)

Discipline (§W1):
  * Detect frozen vs source correctly.
  * Resolve installed resources without depending on CWD.
  * Resolve writable app-data separately from installed files.
  * Avoid importing the full training stack merely to display the launcher.
  * Return explicit exit codes.
  * Write structured error records.
  * Never execute generated command strings; never pass control to a shell.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

from aeon.windows_paths import (
    is_frozen, installed_resource_root, user_data_root, ensure_writable_layout,
    resolve_installed, logs_dir, config_dir,
)


# ---------------------------------------------------------------------------
# Exit codes (stable — parsed by the launcher and any external orchestrator)
# ---------------------------------------------------------------------------
EXIT_OK = 0
EXIT_USER_ARG_ERROR = 2
EXIT_INSTALLATION_INVALID = 3
EXIT_CONFIG_INVALID = 4
EXIT_JOB_NOT_FOUND = 5
EXIT_CHECKPOINT_NOT_FOUND = 6
EXIT_INTEGRITY_FAILURE = 7
EXIT_WORKER_FAILED = 8
EXIT_INTERNAL_ERROR = 99


# ---------------------------------------------------------------------------
# Structured error record
# ---------------------------------------------------------------------------
def _write_error_record(kind: str, exit_code: int, detail: str,
                         args: Optional[List[str]] = None) -> None:
    """Best-effort structured error log in <logs>/errors.jsonl. Never raises."""
    try:
        logs_dir().mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "kind": kind, "exit_code": int(exit_code),
                "detail": detail, "args": args or []}
        with open(logs_dir() / "errors.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Build the dispatch parser. Mutually-exclusive internal modes; default
    (no args) launches the GUI."""
    p = argparse.ArgumentParser(
        prog="Aeon",
        description="Aeon defensive-resilience runtime (Aeon.exe)",
        add_help=True,
    )
    modes = p.add_mutually_exclusive_group()
    modes.add_argument("--worker", metavar="JOB_FILE",
                        help="internal: run the training worker for JOB_FILE")
    modes.add_argument("--verify-installation", action="store_true",
                        help="verify the installed runtime manifest and exit")
    modes.add_argument("--validate-config", metavar="CONFIG_FILE",
                        help="validate a config file's schema + preflight + exit")
    modes.add_argument("--diagnose", metavar="CHECKPOINT",
                        help="internal: offline diagnostics on CHECKPOINT")
    modes.add_argument("--recover", metavar="RECOVERY_REQUEST",
                        help="internal: authorised recovery from RECOVERY_REQUEST")
    p.add_argument("--version", action="store_true", help="print version + exit")
    return p


# ---------------------------------------------------------------------------
# Dispatch handlers — kept LAZY on heavy imports (torch, GUI) so a --verify-
# installation or --version invocation does not pull in the training stack.
# ---------------------------------------------------------------------------
def _dispatch_gui() -> int:
    """Launch the desktop launcher (W2). Ensures writable layout exists."""
    try:
        ensure_writable_layout()
        from aeon.launcher.gui import run_launcher
    except Exception as e:
        _write_error_record("gui_import_failure", EXIT_INTERNAL_ERROR, str(e))
        print(f"aeon: launcher unavailable ({e})", file=sys.stderr)
        return EXIT_INTERNAL_ERROR
    return run_launcher()


def _dispatch_worker(job_file: str) -> int:
    """Run a training worker for the given job.json (W3). This is the ONLY
    dispatch that imports torch — the GUI must remain lightweight."""
    try:
        from aeon.job.worker import run_worker           # heavy import — worker only
    except Exception as e:
        _write_error_record("worker_import_failure", EXIT_INTERNAL_ERROR,
                             str(e), [job_file])
        print(f"aeon: worker unavailable ({e})", file=sys.stderr)
        return EXIT_INTERNAL_ERROR
    try:
        return int(run_worker(job_file))
    except SystemExit as se:
        return int(se.code or 0)
    except Exception as e:
        _write_error_record("worker_crashed", EXIT_WORKER_FAILED, repr(e), [job_file])
        return EXIT_WORKER_FAILED


def _dispatch_verify_installation() -> int:
    """Verify the installed runtime manifest (W5/W7). Lightweight — no torch."""
    try:
        from aeon.integrity import verify_installed_manifest
    except Exception as e:
        _write_error_record("verify_import_failure", EXIT_INTERNAL_ERROR, str(e))
        print(f"aeon: verifier unavailable ({e})", file=sys.stderr)
        return EXIT_INTERNAL_ERROR
    ok, report = verify_installed_manifest()
    print(json.dumps(report, indent=2, sort_keys=True))
    return EXIT_OK if ok else EXIT_INTEGRITY_FAILURE


def _dispatch_validate_config(config_path: str) -> int:
    """Load + schema-check a YAML config; run preflight; return exit code."""
    if not os.path.exists(config_path):
        print(f"aeon: config not found: {config_path}", file=sys.stderr)
        return EXIT_CONFIG_INVALID
    try:
        from aeon.config.schema import validate_config_file
    except Exception as e:
        _write_error_record("validator_import_failure", EXIT_INTERNAL_ERROR,
                             str(e), [config_path])
        print(f"aeon: validator unavailable ({e})", file=sys.stderr)
        return EXIT_INTERNAL_ERROR
    errors = validate_config_file(config_path)
    if errors:
        print(json.dumps({"valid": False, "errors": errors}, indent=2))
        return EXIT_CONFIG_INVALID
    print(json.dumps({"valid": True, "config": os.path.abspath(config_path)},
                      indent=2))
    return EXIT_OK


def _dispatch_diagnose(checkpoint: str) -> int:
    if not os.path.exists(checkpoint):
        print(f"aeon: checkpoint not found: {checkpoint}", file=sys.stderr)
        return EXIT_CHECKPOINT_NOT_FOUND
    try:
        # Reuse scripts/diagnose.py's functions
        sys.path.insert(0, str(resolve_installed("scripts")))
        import diagnose as diag_module            # source or bundled scripts dir
    except Exception as e:
        _write_error_record("diagnose_import_failure", EXIT_INTERNAL_ERROR,
                             str(e), [checkpoint])
        print(f"aeon: diagnostics unavailable ({e})", file=sys.stderr)
        return EXIT_INTERNAL_ERROR
    # Diagnose's CLI: --config + --ckpt + --subcommand. The launcher passes a
    # prepared invocation via its own diagnostic manager; direct dispatch here
    # is a low-level fallback used by scripts.
    sys.argv = ["diagnose", "--config", str(resolve_installed("configs/aeon_smoke_e5.yaml")),
                 "--ckpt", checkpoint, "--subcommand", "certificate"]
    try:
        diag_module.main()
        return EXIT_OK
    except SystemExit as se:
        return int(se.code or 0)


def _dispatch_recover(request_path: str) -> int:
    """Apply an operator-authorised recovery. Requires a well-formed
    RecoveryDecision JSON — refuses without it (F3.3 rule)."""
    if not os.path.exists(request_path):
        print(f"aeon: recovery request not found: {request_path}", file=sys.stderr)
        return EXIT_USER_ARG_ERROR
    try:
        from aeon.protected_checkpoint import RecoveryDecision
    except Exception as e:
        _write_error_record("recover_import_failure", EXIT_INTERNAL_ERROR,
                             str(e), [request_path])
        return EXIT_INTERNAL_ERROR
    try:
        payload = json.load(open(request_path, encoding="utf-8"))
        RecoveryDecision(**payload)                    # dataclass constructor validates
    except Exception as e:
        print(f"aeon: recovery request invalid: {e}", file=sys.stderr)
        return EXIT_USER_ARG_ERROR
    # The actual apply-recovery flow is orchestrated by the launcher which
    # authenticates the target checkpoint via protected_load + this decision.
    print(json.dumps({"recovery_request_valid": True,
                       "next_step": "launcher applies decision at protected_load time"}))
    return EXIT_OK


def _dispatch_version() -> int:
    from aeon.version import RELEASE_METADATA
    print(json.dumps(RELEASE_METADATA, indent=2, sort_keys=True))
    return EXIT_OK


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as se:                            # argparse's exit
        return int(se.code or EXIT_USER_ARG_ERROR)

    if args.version:
        return _dispatch_version()
    if args.worker is not None:
        return _dispatch_worker(args.worker)
    if args.verify_installation:
        return _dispatch_verify_installation()
    if args.validate_config is not None:
        return _dispatch_validate_config(args.validate_config)
    if args.diagnose is not None:
        return _dispatch_diagnose(args.diagnose)
    if args.recover is not None:
        return _dispatch_recover(args.recover)
    return _dispatch_gui()


if __name__ == "__main__":
    # PyInstaller on Windows requires this to be the FIRST executable statement
    # in the frozen entry point. Without it, any child process (torch DataLoader
    # workers, autograd hooks that spawn) re-enters Aeon.exe recursively at
    # spawn time instead of running the target function, and the worker exits
    # with a bare non-zero rc and no stderr (windowed subsystem swallows the
    # spawn-time crash). Cheap no-op in source mode.
    import multiprocessing
    multiprocessing.freeze_support()
    sys.exit(main())
