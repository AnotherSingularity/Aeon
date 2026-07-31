"""aeon/launcher/gui.py — Tkinter desktop launcher.

Section §W2 requirements: display installation status, training status, live
info, controls. Closing the launcher must NOT kill a running worker; a valid
running worker must be reattached on launcher restart. All heavy imports
(torch, PyInstaller machinery) are avoided here — this module is used by both
frozen and source runs to display the launcher window.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from aeon.launcher.controls import (
    acquire_launcher_lock, reattach_or_mark_interrupted, spawn_worker,
    controls_gate,
)
from aeon.job.manager import (
    Job, JobStatus, active_jobs, safe_stop_request, request_emergency_terminate,
    is_stop_requested,
)
from aeon.job.identity import verify_worker_identity
from aeon.windows_paths import (
    user_data_root, logs_dir, evidence_dir, config_dir,
    default_checkpoint_dir,
)
from aeon.version import RELEASE_METADATA
from aeon.config.schema import load_user_config, atomic_write_user_config
from aeon.config.preflight import run_preflight, PreflightVerdict


def run_launcher() -> int:
    """Entry point invoked by aeon.entry._dispatch_gui."""
    try:
        import tkinter as tk
        from tkinter import ttk, filedialog, messagebox
    except Exception as e:
        # tkinter should ship with the frozen bundle. If it doesn't,
        # fall back to a clear stderr message.
        print(f"aeon: Tkinter unavailable ({e}); launcher cannot start", file=sys.stderr)
        return 99
    try:
        lock = acquire_launcher_lock()
    except Exception as e:
        print(f"aeon: {e}", file=sys.stderr)
        return 3
    try:
        app = LauncherApp(tk, ttk, filedialog, messagebox)
        app.mainloop()
        return 0
    finally:
        lock.release()


class LauncherApp:
    """Thin Tk view. Keeps polling job status in a background thread and
    marshals updates to the Tk mainloop via `after()`. No blocking I/O in the
    UI thread."""

    def __init__(self, tk, ttk, filedialog, messagebox):
        self.tk = tk; self.ttk = ttk
        self.filedialog = filedialog; self.messagebox = messagebox

        self.root = tk.Tk()
        self.root.title(f"Aeon — {RELEASE_METADATA.get('semantic_version', 'dev')}")
        self.root.geometry("640x520")

        # State snapshot
        self.state: Dict[str, Any] = {
            "installation_verified": False,
            "preflight_verdict": None,
            "job": None,             # currently attached Job
            "job_status": None,      # current status string
            "live": {},
            "last_error": None,
        }
        self._q: "queue.Queue" = queue.Queue()
        self._build_ui()
        self._start_poll_thread()
        self._reattach()

    # ---- UI layout --------------------------------------------------------
    def _build_ui(self) -> None:
        tk, ttk = self.tk, self.ttk

        # Top: installation panel
        top = ttk.LabelFrame(self.root, text="Installation")
        top.pack(fill="x", padx=10, pady=6)
        self.v_release = tk.StringVar(value=self._release_line())
        self.v_install = tk.StringVar(value="Integrity: not yet verified")
        ttk.Label(top, textvariable=self.v_release).pack(anchor="w", padx=6, pady=1)
        ttk.Label(top, textvariable=self.v_install).pack(anchor="w", padx=6, pady=1)

        # Middle: training status
        mid = ttk.LabelFrame(self.root, text="Training")
        mid.pack(fill="both", expand=True, padx=10, pady=6)
        self.v_status = tk.StringVar(value="Status: idle")
        self.v_live = tk.StringVar(value="")
        ttk.Label(mid, textvariable=self.v_status).pack(anchor="w", padx=6, pady=2)
        ttk.Label(mid, textvariable=self.v_live, justify="left").pack(anchor="w", padx=6, pady=2)

        # Bottom: controls
        bot = ttk.Frame(self.root)
        bot.pack(fill="x", padx=10, pady=8)
        self._buttons: Dict[str, "ttk.Button"] = {}
        controls_layout = [
            ("configure",           "Configure",              self._on_configure),
            ("verify_installation", "Verify installation",    self._on_verify),
            ("run_preflight",       "Run preflight",          self._on_preflight),
            ("start_new_training",  "Start Training",         self._on_start),
            ("resume_latest",       "Resume latest",          self._on_resume),
            ("stop_safely",         "Stop Safely",            self._on_stop_safe),
            ("emergency_stop",      "Emergency Stop",         self._on_stop_emergency),
            ("validate",            "Validate",               self._on_validate),
            ("diagnose_checkpoint", "Diagnose checkpoint",    self._on_diagnose),
            ("open_logs",           "Open logs",              lambda: self._reveal(logs_dir())),
            ("open_checkpoints",    "Open checkpoints",       lambda: self._reveal(default_checkpoint_dir())),
            ("open_evidence",       "Open evidence",          lambda: self._reveal(evidence_dir())),
            ("recovery",            "Recovery",               self._on_recovery),
            ("exit_launcher",       "Exit",                   self._on_exit),
        ]
        col, row = 0, 0
        for key, label, cb in controls_layout:
            b = ttk.Button(bot, text=label, command=cb)
            b.grid(row=row, column=col, padx=3, pady=3, sticky="ew")
            self._buttons[key] = b
            col += 1
            if col >= 4:
                col, row = 0, row + 1
        for c in range(4):
            bot.columnconfigure(c, weight=1)

        # Distinguish emergency stop visually
        try:
            self._buttons["emergency_stop"].configure(style="Emergency.TButton")
        except Exception:
            pass

        self._apply_gates()

    def _release_line(self) -> str:
        r = RELEASE_METADATA
        return (f"Aeon {r.get('semantic_version', '?')}  commit={r.get('source_commit', '?')[:8]}  "
                f"build={r.get('build_type', 'development')}  "
                f"signed={'yes' if r.get('signed') else 'no'}")

    # ---- gating -----------------------------------------------------------
    def _apply_gates(self) -> None:
        gates = controls_gate(self.state["job_status"],
                               self.state["installation_verified"],
                               self.state["preflight_verdict"])
        for k, enabled in gates.items():
            btn = self._buttons.get(k)
            if btn is None: continue
            btn.state(["!disabled" if enabled else "disabled"])

    # ---- polling thread ---------------------------------------------------
    def _start_poll_thread(self) -> None:
        stop_flag = threading.Event()
        self._poll_stop = stop_flag

        def loop():
            while not stop_flag.is_set():
                try:
                    self._poll_once()
                except Exception:
                    pass
                stop_flag.wait(0.5)
        t = threading.Thread(target=loop, name="aeon-launcher-poll", daemon=True)
        t.start()
        self._poll_thread = t
        self.root.after(200, self._drain_queue)

    def _poll_once(self) -> None:
        job = self.state.get("job")
        if job is None:
            return
        # Read status.json + latest metrics
        try:
            status = None
            if os.path.exists(job.status_json_path):
                status = json.load(open(job.status_json_path, encoding="utf-8"))
        except Exception:
            status = None
        # Read the last few metrics lines
        live: Dict[str, Any] = {}
        try:
            metrics_path = Path(job.metrics_dir) / "metrics.jsonl"
            if metrics_path.exists():
                with open(metrics_path, encoding="utf-8") as fh:
                    last_lines = fh.readlines()[-8:]
                for line in last_lines:
                    try:
                        r = json.loads(line)
                        if r.get("kind") == "always_on":
                            live = {"step": r.get("step"), "loss": r.get("loss"),
                                     "lr": r.get("lr"), "tps": r.get("tokens_per_s_raw"),
                                     "mem_mb": r.get("resident_mb"),
                                     "cert": r.get("certificate_holds"),
                                     "gamma": r.get("gamma")}
                    except Exception:
                        continue
        except Exception:
            pass
        self._q.put(("status", status, live))

    def _drain_queue(self) -> None:
        try:
            while True:
                kind, *payload = self._q.get_nowait()
                if kind == "status":
                    status, live = payload
                    if status is not None:
                        self.state["job_status"] = status.get("status")
                    if live:
                        self.state["live"] = live
                    self._render_status()
                    self._apply_gates()
        except queue.Empty:
            pass
        self.root.after(300, self._drain_queue)

    def _render_status(self) -> None:
        js = self.state.get("job_status") or "idle"
        self.v_status.set(f"Status: {js}")
        live = self.state.get("live") or {}
        lines = []
        for k in ("step", "loss", "lr", "tps", "mem_mb", "cert", "gamma"):
            if k in live and live[k] is not None:
                lines.append(f"  {k}: {live[k]}")
        self.v_live.set("\n".join(lines) or "  (waiting for worker metrics)")

    # ---- reattachment on startup -----------------------------------------
    def _reattach(self) -> None:
        for job, ident in reattach_or_mark_interrupted():
            if ident is not None:
                self.state["job"] = job
                self.state["job_status"] = "RUNNING"
                self._render_status()
                break

    # ---- button handlers --------------------------------------------------
    def _on_configure(self):
        cfg_path = str(config_dir() / "user_config.json")
        try:
            current = load_user_config(cfg_path) or {}
            # Simple dialog: ask for tokenizer + corpus + checkpoint dir
            tok = self.filedialog.askopenfilename(
                title="Select tokenizer (.model)", initialdir=current.get("tokenizer_path", ""))
            if not tok: return
            cor = self.filedialog.askdirectory(
                title="Select corpus directory", initialdir=current.get("corpus_path", ""))
            if not cor: return
            ck = self.filedialog.askdirectory(
                title="Select checkpoint directory",
                initialdir=current.get("checkpoint_dir", str(default_checkpoint_dir())))
            if not ck: return
            current.update({"tokenizer_path": tok, "corpus_path": cor,
                             "checkpoint_dir": ck,
                             "metrics_dir": str(user_data_root() / "metrics"),
                             "evidence_dir": str(evidence_dir()),
                             "training_config_id": "aeon_350m_primary.yaml",
                             "resume_preference": "auto",
                             "checkpoint_interval": 1000,
                             "validation_interval": 1000,
                             "disk_allocation_gb": 32,
                             "cpu_thread_limit": max(1, (os.cpu_count() or 1) - 1)})
            atomic_write_user_config(cfg_path, current)
            self.messagebox.showinfo("Aeon", f"Configuration written to {cfg_path}")
        except Exception as e:
            self.messagebox.showerror("Aeon", f"configure failed: {e}")

    def _on_verify(self):
        from aeon.integrity import verify_installed_manifest
        ok, report = verify_installed_manifest()
        self.state["installation_verified"] = bool(ok)
        self.v_install.set(f"Integrity: {'VERIFIED' if ok else 'FAILED'}  "
                            f"({report.get('files_ok', 0)} files ok, "
                            f"{len(report.get('mismatched', []))} mismatched, "
                            f"{len(report.get('missing', []))} missing)")
        self._apply_gates()

    def _on_preflight(self):
        cfg = load_user_config(str(config_dir() / "user_config.json")) or {}
        res = run_preflight(cfg)
        self.state["preflight_verdict"] = res.verdict.value
        self._apply_gates()
        self.messagebox.showinfo(
            "Preflight", f"{res.verdict.value}\n\n" +
            "\n".join(f"[{c.status}] {c.name}: {c.detail}" for c in res.checks))

    # W10-3: distinct Start / Resume / Recovery paths. Each creates a Job
    # with a different `intent` field, spawns a fresh worker, and writes a
    # distinct audit event under audit_dir/launcher_events.jsonl.

    def _emit_launcher_event(self, kind: str, payload: dict) -> None:
        """Best-effort structured event log — never raises."""
        try:
            from aeon.windows_paths import evidence_dir as evd
            import time as _t
            path = os.path.join(str(evd()), "launcher_events.jsonl")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": _t.time(), "kind": kind, **payload},
                                       sort_keys=True) + "\n")
        except Exception:
            pass

    def _on_start(self):
        """Fresh training. Refuses if the target checkpoint_dir already
        contains an authenticated chain — that's the audit's "no
        accidental overwrite" gate. The worker enforces the same rule
        server-side."""
        from aeon.job.manager import create_job
        from aeon.windows_paths import evidence_dir as evd, resolve_installed
        cfg = load_user_config(str(config_dir() / "user_config.json")) or {}
        tcid = cfg.get("training_config_id", "aeon_350m_primary.yaml")
        try:
            job = create_job(
                config_path=str(resolve_installed(f"configs/{tcid}")),
                tokenizer_path=cfg.get("tokenizer_path"),
                corpus_path=cfg.get("corpus_path"),
                checkpoint_dir=cfg.get("checkpoint_dir", str(default_checkpoint_dir())),
                metrics_dir=cfg.get("metrics_dir", str(user_data_root() / "metrics")),
                audit_dir=cfg.get("evidence_dir", str(evd())),
                checkpoint_policy={"interval": int(cfg.get("checkpoint_interval", 1000))},
                intent="start",
            )
            spawn_worker(job)
            self.state["job"] = job; self.state["job_status"] = "STARTING"
            self._emit_launcher_event("start_new_training",
                {"job_id": job.job_id, "checkpoint_dir": job.checkpoint_dir})
            # Persist the job dir so Resume can locate the HMAC key later.
            cfg["last_job_dir"] = job.job_dir
            atomic_write_user_config(str(config_dir() / "user_config.json"), cfg)
            self._render_status(); self._apply_gates()
        except Exception as e:
            self.messagebox.showerror("Aeon", f"start failed: {e}")

    def _on_resume(self):
        """Resume Latest. Enumerates authenticated checkpoints under the
        configured checkpoint_dir, picks the newest one, and spawns a
        worker with intent='resume'. Refuses (with an explicit message)
        if no authenticated checkpoint exists — the button gate in
        _apply_gates should already prevent this path, but the runtime
        check is defensive.

        The worker's protected_load enforces MAC / anti-rollback /
        schema/K/vocab gates in one call; failure surfaces as a FAILED
        job status with a structured reason. Resume DOES NOT alias to
        Start (W10-3 flip of audit finding A6)."""
        from aeon.job.manager import create_job
        from aeon.job.key_store import ensure_job_hmac_keyref
        from aeon.launcher.resume import latest_authenticated_checkpoint
        from aeon.windows_paths import evidence_dir as evd, resolve_installed

        cfg = load_user_config(str(config_dir() / "user_config.json")) or {}
        ckpt_dir = cfg.get("checkpoint_dir", str(default_checkpoint_dir()))

        # Enumerate under a temp keyref BOUND TO NO JOB YET — but we can
        # only authenticate the checkpoints if we have the ORIGINAL job's
        # key, which lives at that job's job_dir/hmac.key. Convention: the
        # launcher stores a pointer from checkpoint_dir back to job_dir in
        # cfg["last_job_dir"], which _on_start writes after spawning. If
        # the pointer is absent, the launcher must ask the user (owned by
        # W10-9's polish); for W10-3 we tell them plainly and refuse.
        last_job_dir = cfg.get("last_job_dir")
        if not last_job_dir or not os.path.isdir(last_job_dir):
            self.messagebox.showerror("Aeon",
                "Resume requires a known-good previous job dir under "
                f"{ckpt_dir!r}. None recorded in user_config.last_job_dir. "
                "Start a new training run, or select a job dir via "
                "Recovery (W10-9 will surface an authenticated picker).")
            return

        try:
            keyref = ensure_job_hmac_keyref(last_job_dir, allow_create=False)
        except Exception as e:
            self.messagebox.showerror("Aeon", f"Resume: HMAC key unavailable: {e}")
            return

        cand = latest_authenticated_checkpoint(ckpt_dir, keyref)
        if cand is None:
            self.messagebox.showerror("Aeon",
                f"Resume refused: no authenticated checkpoint under {ckpt_dir!r}.")
            return

        tcid = cfg.get("training_config_id", "aeon_350m_primary.yaml")
        try:
            job = create_job(
                config_path=str(resolve_installed(f"configs/{tcid}")),
                tokenizer_path=cfg.get("tokenizer_path"),
                corpus_path=cfg.get("corpus_path"),
                checkpoint_dir=ckpt_dir,
                metrics_dir=cfg.get("metrics_dir", str(user_data_root() / "metrics")),
                audit_dir=cfg.get("evidence_dir", str(evd())),
                checkpoint_policy={"interval": int(cfg.get("checkpoint_interval", 1000))},
                intent="resume",
                resume_from_checkpoint=cand.path,
            )
            # Copy the HMAC key from the previous job so the new worker can
            # authenticate. W10-6 replaces this with a signed manifest root.
            import shutil as _shutil
            _shutil.copy(os.path.join(last_job_dir, "hmac.key"),
                          os.path.join(job.job_dir, "hmac.key"))
            spawn_worker(job)
            self.state["job"] = job; self.state["job_status"] = "STARTING"
            self._emit_launcher_event("resume_latest",
                {"job_id": job.job_id,
                  "resume_from": cand.path,
                  "resume_step": cand.step,
                  "authorized_step": cand.authorized_step,
                  "source_job_dir": last_job_dir})
            self._render_status(); self._apply_gates()
        except Exception as e:
            self.messagebox.showerror("Aeon", f"Resume failed: {e}")

    def _on_stop_safe(self):
        j = self.state.get("job")
        if j is None: return
        safe_stop_request(j)
        self.messagebox.showinfo("Aeon",
            "Safe stop requested. The worker will finish the current window, "
            "save an authenticated checkpoint, and exit.")

    def _on_stop_emergency(self):
        j = self.state.get("job")
        if j is None: return
        if not self.messagebox.askyesno(
                "Aeon — Emergency Stop",
                "This aborts the worker without a final checkpoint. UNSAVED "
                "PROGRESS SINCE THE LAST CHECKPOINT MAY BE LOST. Continue?"):
            return
        request_emergency_terminate(j)

    def _on_validate(self):
        """W10-9/A18: real Validate. Enumerate authenticated checkpoints
        under the configured checkpoint_dir, pick the latest one, run the
        offline diagnostic subprocess (Aeon.exe --diagnose <ck>) with
        stdout/stderr captured, and display the report in a modal text
        window. Bounded: subprocess.run with a 300 s timeout so the GUI
        thread never hangs indefinitely.

        Also surfaces the most recent periodic in-worker validation lines
        recorded by _run_periodic_validation (worker.py, W10-9/A7).
        """
        cfg = load_user_config(str(config_dir() / "user_config.json")) or {}
        ck_dir = cfg.get("checkpoint_dir") or str(default_checkpoint_dir())
        try:
            from aeon.launcher.resume import latest_authenticated_checkpoint
            keyref = self._resolve_keyref(cfg)
            latest = latest_authenticated_checkpoint(ck_dir, keyref)
        except Exception as e:
            self.messagebox.showerror("Aeon",
                f"Validate: cannot enumerate checkpoints under {ck_dir}: {e}")
            return
        if latest is None:
            self.messagebox.showinfo("Aeon",
                "Validate: no authenticated checkpoint found in "
                f"{ck_dir}. Start a training run first.")
            return
        # Run --diagnose synchronously with captured output.
        try:
            r = subprocess.run(
                [sys.executable, "-m", "aeon.entry", "--diagnose", latest.path],
                capture_output=True, text=True, timeout=300)
            report = (r.stdout or "") + ("\n[stderr]\n" + r.stderr if r.stderr else "")
            title = f"Validate — {os.path.basename(latest.path)}"
        except subprocess.TimeoutExpired:
            report = "[Validate] diagnostic timed out (>300 s)"
            title = "Validate — timed out"
        except Exception as e:
            self.messagebox.showerror("Aeon", f"Validate spawn failed: {e}")
            return
        # Include periodic worker validation tail
        worker_val = self._tail_worker_validation(cfg, n=5)
        if worker_val:
            report += "\n\n[recent periodic worker validation]\n" + worker_val
        self._show_scrollable_report(title, report or "(no output)")

    def _on_diagnose(self):
        """W10-9/A20: Diagnose runs the subprocess with captured stdout/stderr
        and displays the report. Output used to be silently discarded, which
        gave the operator no visible signal that anything ran."""
        ck = self.filedialog.askopenfilename(title="Select checkpoint to diagnose")
        if not ck: return
        try:
            r = subprocess.run(
                [sys.executable, "-m", "aeon.entry", "--diagnose", ck],
                capture_output=True, text=True, timeout=300)
            report = (r.stdout or "") + ("\n[stderr]\n" + r.stderr if r.stderr else "")
            self._show_scrollable_report(
                f"Diagnose — {os.path.basename(ck)}", report or "(no output)")
        except subprocess.TimeoutExpired:
            self.messagebox.showerror("Aeon", "diagnose timed out (>300 s)")
        except Exception as e:
            self.messagebox.showerror("Aeon", f"diagnose failed: {e}")

    def _resolve_keyref(self, cfg):
        """Best-effort resolution of the last-known job's HMAC keyref for
        checkpoint enumeration. Recovery/Validate need it to authenticate
        stored generations. If no last job is known, returns None; the
        enumerator downgrades to unverified metadata."""
        last_job_dir = cfg.get("last_job_dir")
        if not last_job_dir or not os.path.isdir(last_job_dir):
            return None
        try:
            from aeon.job.key_store import ensure_job_hmac_keyref
            return ensure_job_hmac_keyref(last_job_dir, allow_create=False)
        except Exception:
            return None

    def _tail_worker_validation(self, cfg, n: int = 5) -> str:
        """W10-9/A7: read the tail of worker_validation.jsonl so Validate
        can show the freshest periodic-in-worker validations without
        re-running the diagnostic loop."""
        try:
            from aeon.windows_paths import evidence_dir as evd
            path = os.path.join(cfg.get("evidence_dir", str(evd())),
                                 "worker_validation.jsonl")
        except Exception:
            return ""
        if not os.path.exists(path):
            return ""
        try:
            with open(path, encoding="utf-8") as fh:
                lines = fh.readlines()[-n:]
            return "".join(lines).rstrip()
        except Exception:
            return ""

    def _show_scrollable_report(self, title: str, text: str) -> None:
        """Modal scrollable text window (W10-9). Kept lightweight — tkinter
        Text + Scrollbar. Falls back to messagebox.showinfo when tkinter
        isn't available (headless tests)."""
        try:
            import tkinter as tk
            from tkinter import scrolledtext
            top = tk.Toplevel(self.root)
            top.title(title)
            top.geometry("880x520")
            box = scrolledtext.ScrolledText(top, wrap="none", font=("Consolas", 9))
            box.pack(fill="both", expand=True)
            box.insert("1.0", text)
            box.configure(state="disabled")
            tk.Button(top, text="Close", command=top.destroy).pack(pady=4)
        except Exception:
            self.messagebox.showinfo(title, text[:4000])

    def _on_recovery(self):
        """W10-9/A19: interactive Recovery. Instead of demanding a
        pre-built operator-signed RecoveryDecision JSON file (the
        launcher previously "instructed the user to open a terminal"),
        the GUI now:

          1. Enumerates authenticated checkpoint candidates under the
             configured checkpoint_dir via aeon.launcher.resume.
          2. Presents them in a modal picker (step + timestamp + path).
          3. Prompts the operator for the anti-rollback bypass reason.
          4. Constructs the RecoveryDecision in-process and writes it
             into the new job dir.
          5. Spawns a worker with intent="recover".

        The RecoveryDecision is still recorded on disk under the new
        job dir for evidence and audit — the change is that the GUI
        builds it, not the operator with a text editor."""
        from aeon.job.manager import create_job
        from aeon.windows_paths import evidence_dir as evd, resolve_installed
        from aeon.launcher.resume import enumerate_checkpoints
        cfg = load_user_config(str(config_dir() / "user_config.json")) or {}
        last_job_dir = cfg.get("last_job_dir")
        if not last_job_dir or not os.path.isdir(last_job_dir):
            self.messagebox.showerror("Aeon",
                "Recovery requires a known-good previous job dir. None "
                "recorded in user_config.last_job_dir.")
            return
        ck_dir = cfg.get("checkpoint_dir", str(default_checkpoint_dir()))
        keyref = self._resolve_keyref(cfg)
        try:
            candidates = enumerate_checkpoints(ck_dir, keyref)
        except Exception as e:
            self.messagebox.showerror("Aeon",
                f"Recovery: cannot enumerate {ck_dir}: {e}")
            return
        candidates = [c for c in candidates if getattr(c, "authenticated", False)]
        if not candidates:
            self.messagebox.showerror("Aeon",
                f"Recovery: no authenticated checkpoints under {ck_dir}.")
            return
        chosen = self._pick_recovery_candidate(candidates)
        if chosen is None:
            return
        reason = self._prompt_recovery_reason()
        if not reason:
            return
        # Build the RecoveryDecision in-process via BuildableRecoveryDecision
        # which fills in the fields the operator would otherwise have to type.
        from aeon.launcher.resume import (
            BuildableRecoveryDecision, latest_authenticated_checkpoint,
        )
        current = latest_authenticated_checkpoint(ck_dir, keyref)
        current_state_identity = (
            f"sha256:{current.mac_algo}:step={current.step}" if current is not None
            else "sha256:absent")
        _t = time.time()
        brd = BuildableRecoveryDecision(
            candidate=chosen, reason=reason,
            operator_authorization_ref=f"desktop_operator:{int(_t)}",
            current_state_identity=current_state_identity)
        rd = brd.build()
        tcid = cfg.get("training_config_id", "aeon_350m_primary.yaml")
        try:
            job = create_job(
                config_path=str(resolve_installed(f"configs/{tcid}")),
                tokenizer_path=cfg.get("tokenizer_path"),
                corpus_path=cfg.get("corpus_path"),
                checkpoint_dir=ck_dir,
                metrics_dir=cfg.get("metrics_dir", str(user_data_root() / "metrics")),
                audit_dir=cfg.get("evidence_dir", str(evd())),
                checkpoint_policy={"interval": int(cfg.get("checkpoint_interval", 1000)),
                                     "validation_interval": int(cfg.get("validation_interval", 1000))},
                compute_policy={
                    "cpu_thread_limit": int(cfg.get("cpu_thread_limit", 0) or 0) or None,
                    "memory_ceiling_gb": cfg.get("memory_ceiling_gb"),
                    "resume_preference": cfg.get("resume_preference", "auto"),
                },
                intent="recover",
                resume_from_checkpoint=chosen.path,
            )
            # Write the RecoveryDecision under the new job dir and point
            # the job at it.
            rd_path = os.path.join(job.job_dir, "recovery_decision.json")
            with open(rd_path, "w", encoding="utf-8") as fh:
                json.dump(rd.__dict__, fh, sort_keys=True, indent=2)
            job.recovery_decision_path = rd_path
            from aeon.job.manager import _atomic_write_json
            _atomic_write_json(job.job_json_path, job.to_dict())
            import shutil as _shutil
            _shutil.copy(os.path.join(last_job_dir, "hmac.key"),
                          os.path.join(job.job_dir, "hmac.key"))
            spawn_worker(job)
            self.state["job"] = job; self.state["job_status"] = "STARTING"
            self._emit_launcher_event("recovery_authorized",
                {"job_id": job.job_id,
                  "checkpoint": chosen.path,
                  "recovery_decision": rd_path,
                  "resulting_authorized_state": rd.resulting_authorized_state,
                  "reason": rd.reason,
                  "source_job_dir": last_job_dir})
            self._render_status(); self._apply_gates()
        except Exception as e:
            self.messagebox.showerror("Aeon", f"Recovery failed: {e}")

    def _pick_recovery_candidate(self, candidates):
        """Modal listbox for choosing among authenticated candidates.
        Falls back to picking the newest one when running headless."""
        try:
            import tkinter as tk
            top = tk.Toplevel(self.root)
            top.title("Select a checkpoint to recover from")
            top.geometry("640x360")
            lb = tk.Listbox(top, font=("Consolas", 9))
            lb.pack(fill="both", expand=True)
            for i, c in enumerate(candidates):
                label = f"step {getattr(c, 'step', '?'):>10}  {c.state_path}"
                lb.insert(tk.END, label)
            selected = {"i": None}
            def _accept():
                sel = lb.curselection()
                if sel:
                    selected["i"] = sel[0]
                top.destroy()
            tk.Button(top, text="Recover from this checkpoint",
                        command=_accept).pack(pady=4)
            self.root.wait_window(top)
            i = selected["i"]
            return candidates[i] if i is not None else None
        except Exception:
            # Headless fallback — pick the most recent.
            return max(candidates, key=lambda c: getattr(c, "step", 0))

    def _prompt_recovery_reason(self):
        """Modal reason prompt. Falls back to a synthesized default when
        running headless so the RecoveryDecision is still well-formed."""
        try:
            import tkinter as tk
            from tkinter import simpledialog
            return simpledialog.askstring(
                "Recovery reason",
                "State a short human-readable reason for this recovery.\n"
                "It is recorded in the RecoveryDecision and evidence log.",
                parent=self.root) or ""
        except Exception:
            return "desktop_operator_recovery"

    def _on_exit(self):
        # Closing the launcher must NOT terminate a live worker.
        self.root.destroy()

    def _reveal(self, p: Path) -> None:
        p = Path(p); p.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(str(p))                   # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])     # no shell
            else:
                subprocess.Popen(["xdg-open", str(p)]) # no shell
        except Exception:
            pass

    def mainloop(self) -> None:
        try:
            self.root.mainloop()
        finally:
            try: self._poll_stop.set()
            except Exception: pass
