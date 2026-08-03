"""aeon.desktop.chat_ui — Tkinter chat window for the 7M research preview.

Runs the AeonDesktopRuntime on a background thread and streams
generation events into the UI. Send / Stop / New Session / Clear
Conversation buttons are wired to the runtime. Never blocks the UI
thread on model execution.

Prohibitions:
  * No HTML rendering of model output.
  * No script execution.
  * No automatic link navigation.
  * No fake typing animation.
  * No external network transport.
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Optional

from .protocol import (
    ErrorCode, EventKind, GenerationOptions, RuntimeEvent, RuntimeState,
)
from .runtime import AeonDesktopRuntime, RuntimeError_


CHAT_TITLE = "Aeon Desktop — Research Preview (7M P2 Proxy)"
BANNER_TEXT = (
    "Aeon Desktop — Research Preview\n"
    "Model: AEON-LBC-1 P2 Proxy · Scale: 7M · Runtime: Offline · ACIS: OFF"
)


class ChatUIState:
    """UI-facing state; distinct from runtime state so we can render it
    from the main thread without holding runtime locks."""
    STARTING = "Starting runtime"
    VALIDATING = "Validating release"
    LOADING = "Loading Aeon"
    READY = "Ready"
    GENERATING = "Generating"
    STOPPING = "Stopping"
    OFFLINE = "Offline"
    ERROR = "Error"


def runtime_state_to_ui(state: RuntimeState) -> str:
    return {
        RuntimeState.NOT_STARTED: ChatUIState.STARTING,
        RuntimeState.STARTING: ChatUIState.STARTING,
        RuntimeState.PREFLIGHT: ChatUIState.STARTING,
        RuntimeState.VALIDATING_RELEASE: ChatUIState.VALIDATING,
        RuntimeState.LOADING_MODEL: ChatUIState.LOADING,
        RuntimeState.READY: ChatUIState.READY,
        RuntimeState.GENERATING: ChatUIState.GENERATING,
        RuntimeState.CANCELLING: ChatUIState.STOPPING,
        RuntimeState.FAILED: ChatUIState.ERROR,
        RuntimeState.SHUTTING_DOWN: ChatUIState.STOPPING,
        RuntimeState.STOPPED: ChatUIState.OFFLINE,
    }[state]


class ChatController:
    """Non-Tk controller — owns the runtime + event queue + session id.

    Split from the Tk widgets so it can be exercised by headless tests.
    """

    def __init__(self, release_root: Path):
        self.release_root = Path(release_root)
        self.event_q: "queue.Queue[RuntimeEvent]" = queue.Queue(maxsize=4096)
        self.runtime = AeonDesktopRuntime(event_handler=self._on_event)
        self.session_id: Optional[str] = None
        self.active_request_id: Optional[str] = None
        self.ui_state = ChatUIState.STARTING
        self._load_lock = threading.Lock()

    def _on_event(self, ev: RuntimeEvent) -> None:
        try:
            self.event_q.put_nowait(ev)
        except queue.Full:
            # Drop oldest events on overflow; never block runtime
            try: self.event_q.get_nowait()
            except queue.Empty: pass
            try: self.event_q.put_nowait(ev)
            except queue.Full: pass

    def bootstrap(self) -> None:
        """Runs runtime.preflight() + load_release() on the caller's thread.
        Call this on a background thread from the UI."""
        with self._load_lock:
            self.runtime.preflight()
            self.runtime.load_release(self.release_root)
            self.session_id = self.runtime.create_session()

    def send(self, prompt: str, options: Optional[GenerationOptions] = None) -> str:
        if self.session_id is None:
            raise RuntimeError_(ErrorCode.RUNTIME_START_FAILED, "no session")
        rid = self.runtime.submit_prompt(self.session_id, prompt, options)
        self.active_request_id = rid
        return rid

    def stop(self) -> bool:
        if self.active_request_id is None:
            return False
        ok = self.runtime.cancel(self.active_request_id)
        return ok

    def new_session(self) -> str:
        if self.session_id is not None:
            try: self.runtime.close_session(self.session_id)
            except RuntimeError_: pass
        self.session_id = self.runtime.create_session()
        return self.session_id

    def clear_conversation(self) -> None:
        if self.session_id is not None:
            self.runtime.reset_session(self.session_id)

    def shutdown(self) -> None:
        self.runtime.shutdown()

    def diagnostics(self) -> dict:
        return self.runtime.diagnostics()


def run_chat_ui(release_root: Path) -> int:
    """Main entry point — creates the Tk window and runs the event loop.

    Only imports Tk when actually invoked; the module import itself
    stays UI-toolkit-free so the tests can exercise the controller."""
    import tkinter as tk
    from tkinter import ttk, scrolledtext, messagebox

    ctl = ChatController(Path(release_root))

    root = tk.Tk()
    root.title(CHAT_TITLE)
    root.geometry("800x600")

    # ---------- Layout ----------
    banner = ttk.Label(root, text=BANNER_TEXT, anchor="w", justify="left")
    banner.pack(fill="x", padx=8, pady=(8, 4))

    status_var = tk.StringVar(value=ChatUIState.STARTING)
    status = ttk.Label(root, textvariable=status_var, anchor="w")
    status.pack(fill="x", padx=8)

    transcript = scrolledtext.ScrolledText(root, wrap="word", state="disabled",
                                                 height=18)
    transcript.pack(fill="both", expand=True, padx=8, pady=4)

    prompt_frame = ttk.Frame(root)
    prompt_frame.pack(fill="x", padx=8, pady=4)
    prompt_input = tk.Text(prompt_frame, height=3, wrap="word")
    prompt_input.pack(side="left", fill="x", expand=True)

    btn_frame = ttk.Frame(root)
    btn_frame.pack(fill="x", padx=8, pady=(0, 8))
    send_btn = ttk.Button(btn_frame, text="Send")
    send_btn.pack(side="left")
    stop_btn = ttk.Button(btn_frame, text="Stop", state="disabled")
    stop_btn.pack(side="left", padx=(4, 0))
    new_session_btn = ttk.Button(btn_frame, text="New Session")
    new_session_btn.pack(side="left", padx=(4, 0))
    clear_btn = ttk.Button(btn_frame, text="Clear Conversation")
    clear_btn.pack(side="left", padx=(4, 0))
    diag_btn = ttk.Button(btn_frame, text="Diagnostics")
    diag_btn.pack(side="left", padx=(4, 0))

    def _append(text: str) -> None:
        transcript.configure(state="normal")
        transcript.insert("end", text)
        transcript.see("end")
        transcript.configure(state="disabled")

    def _refresh_button_state() -> None:
        st = ctl.runtime.state()
        status_var.set(runtime_state_to_ui(st))
        if st == RuntimeState.GENERATING:
            send_btn.config(state="disabled")
            stop_btn.config(state="normal")
        elif st in (RuntimeState.CANCELLING,):
            send_btn.config(state="disabled")
            stop_btn.config(state="disabled")
        elif st == RuntimeState.READY:
            send_btn.config(state="normal")
            stop_btn.config(state="disabled")
        else:
            send_btn.config(state="disabled")
            stop_btn.config(state="disabled")

    def _drain_events() -> None:
        drained = 0
        while drained < 32:
            try:
                ev = ctl.event_q.get_nowait()
            except queue.Empty:
                break
            drained += 1
            if ev.event_type is EventKind.TEXT_DELTA:
                _append(ev.payload.get("delta", ""))
            elif ev.event_type is EventKind.GENERATION_STARTED:
                _append("\n> ")
            elif ev.event_type is EventKind.GENERATION_COMPLETED:
                _append("\n\n")
            elif ev.event_type is EventKind.GENERATION_CANCELLED:
                _append("\n[cancelled]\n\n")
            elif ev.event_type is EventKind.GENERATION_FAILED:
                _append(f"\n[failed: {ev.payload.get('code','?')}]\n\n")
            elif ev.event_type is EventKind.RUNTIME_FAILED:
                messagebox.showerror("Aeon runtime failed",
                                          f"{ev.payload.get('code','?')}: "
                                          f"{ev.payload.get('detail','')[:200]}")
        _refresh_button_state()
        root.after(50, _drain_events)

    def _on_send() -> None:
        text = prompt_input.get("1.0", "end").strip()
        if not text:
            return
        _append(f"\n<< {text}\n")
        prompt_input.delete("1.0", "end")
        try:
            ctl.send(text, GenerationOptions())
        except RuntimeError_ as e:
            messagebox.showerror("Cannot send", f"{e.code.value}: {e.detail}")
        _refresh_button_state()

    def _on_stop() -> None:
        ctl.stop()
        _refresh_button_state()

    def _on_new_session() -> None:
        ctl.new_session()
        transcript.configure(state="normal")
        transcript.delete("1.0", "end")
        transcript.configure(state="disabled")

    def _on_clear() -> None:
        ctl.clear_conversation()

    def _on_diag() -> None:
        d = ctl.diagnostics()
        msg = "\n".join(f"{k}: {v}" for k, v in d.items())
        messagebox.showinfo("Aeon diagnostics", msg)

    def _on_close() -> None:
        try: ctl.shutdown()
        except Exception: pass
        root.destroy()

    send_btn.config(command=_on_send)
    stop_btn.config(command=_on_stop)
    new_session_btn.config(command=_on_new_session)
    clear_btn.config(command=_on_clear)
    diag_btn.config(command=_on_diag)
    root.protocol("WM_DELETE_WINDOW", _on_close)

    # Bootstrap on a background thread so the UI stays responsive.
    def _bootstrap():
        try:
            ctl.bootstrap()
        except RuntimeError_ as e:
            root.after(0, lambda: messagebox.showerror(
                "Aeon startup failed", f"{e.code.value}: {e.detail}"))
    threading.Thread(target=_bootstrap, daemon=True).start()

    root.after(50, _drain_events)
    root.mainloop()
    return 0
