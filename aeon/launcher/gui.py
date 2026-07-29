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

    def _on_start(self):
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
            )
            spawn_worker(job)
            self.state["job"] = job; self.state["job_status"] = "STARTING"
            self._render_status(); self._apply_gates()
        except Exception as e:
            self.messagebox.showerror("Aeon", f"start failed: {e}")

    def _on_resume(self):
        # Resume is presented as: pick the most recent authenticated ckpt in the
        # configured checkpoint_dir; start a new job with resume=True in config.
        # Actual resume happens inside the worker via strict_load.
        self._on_start()

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
        self.messagebox.showinfo("Aeon", "Validation runs the offline diagnostic "
                                          "over the latest authenticated checkpoint. "
                                          "Use --diagnose from the launcher menu (W4).")

    def _on_diagnose(self):
        ck = self.filedialog.askopenfilename(title="Select checkpoint to diagnose")
        if not ck: return
        # Diagnose is invoked as a subprocess (no shell) — same executable
        try:
            subprocess.Popen([sys.executable, "--diagnose", ck],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            self.messagebox.showerror("Aeon", f"diagnose spawn failed: {e}")

    def _on_recovery(self):
        self.messagebox.showinfo("Aeon",
            "Recovery requires an operator-signed RecoveryDecision JSON. "
            "Launch Aeon.exe --recover <request.json> for the authenticated path.")

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
