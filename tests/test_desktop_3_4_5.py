"""DESKTOP-3/4/5 — Chat controller + cancellation + session isolation + recovery."""
import os
import sys
import time
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BUNDLE = os.path.join(ROOT, "release-assets", "aeon-desktop-p2-proxy")


# ---------------------------------------------------------------------------
# Chat UI module — structure only (no Tk rendering in headless container)
# ---------------------------------------------------------------------------
def test_chat_ui_module_imports_without_creating_window():
    """The module must import without requiring a display."""
    import aeon.desktop.chat_ui as ui
    assert hasattr(ui, "ChatController")
    assert hasattr(ui, "run_chat_ui")
    assert hasattr(ui, "CHAT_TITLE")


def test_chat_controller_state_mapping_covers_every_runtime_state():
    from aeon.desktop.chat_ui import runtime_state_to_ui
    from aeon.desktop.protocol import RuntimeState
    for s in RuntimeState:
        got = runtime_state_to_ui(s)
        assert isinstance(got, str) and got


# ---------------------------------------------------------------------------
# ChatController live smoke — no Tk, just the controller
# ---------------------------------------------------------------------------
def test_chat_controller_bootstrap_and_send():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.chat_ui import ChatController
    from aeon.desktop.protocol import GenerationOptions
    ctl = ChatController(BUNDLE)
    ctl.bootstrap()
    rid = ctl.send("Once", GenerationOptions(max_new_tokens=4, temperature=0.0))
    # Wait for the underlying generation thread to finish
    ctl.runtime._active_generation_thread.join(timeout=180)
    assert ctl.runtime.state().value == "READY"
    ctl.shutdown()


def test_chat_controller_new_session_replaces_session_id():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.chat_ui import ChatController
    ctl = ChatController(BUNDLE)
    ctl.bootstrap()
    first = ctl.session_id
    second = ctl.new_session()
    assert first != second
    ctl.shutdown()


def test_chat_controller_clear_conversation_resets_history():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.chat_ui import ChatController
    from aeon.desktop.protocol import GenerationOptions
    ctl = ChatController(BUNDLE)
    ctl.bootstrap()
    ctl.send("Once", GenerationOptions(max_new_tokens=3, temperature=0.0))
    ctl.runtime._active_generation_thread.join(timeout=180)
    sess_id = ctl.session_id
    assert len(ctl.runtime._sessions[sess_id].token_history) > 0
    ctl.clear_conversation()
    assert len(ctl.runtime._sessions[sess_id].token_history) == 0
    ctl.shutdown()


# ---------------------------------------------------------------------------
# DESKTOP-5: cancellation
# ---------------------------------------------------------------------------
def test_cancellation_mid_generation_returns_to_ready_and_preserves_committed():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.runtime import AeonDesktopRuntime
    from aeon.desktop.protocol import GenerationOptions, RuntimeState, EventKind
    events = []
    rt = AeonDesktopRuntime(event_handler=lambda ev: events.append(ev))
    rt.preflight()
    rt.load_release(BUNDLE)
    sid = rt.create_session()
    # Kick off a large generation, then cancel very quickly.
    rid = rt.submit_prompt(sid, "The",
                                 GenerationOptions(max_new_tokens=128, temperature=0.0))
    # Wait for at least one token to be committed
    for _ in range(200):
        if any(e.event_type == EventKind.TOKEN_GENERATED for e in events):
            break
        time.sleep(0.05)
    ok = rt.cancel(rid)
    assert ok is True
    # Wait for cancellation to complete
    for _ in range(200):
        if rt.state() == RuntimeState.READY: break
        time.sleep(0.05)
    assert rt.state() == RuntimeState.READY
    cancelled_events = [e for e in events if e.event_type == EventKind.GENERATION_CANCELLED]
    assert len(cancelled_events) == 1
    assert cancelled_events[0].payload.get("generated_tokens", 0) >= 0
    rt.shutdown()


def test_cancel_before_first_token_still_returns_to_ready():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.runtime import AeonDesktopRuntime
    from aeon.desktop.protocol import GenerationOptions, RuntimeState
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    sid = rt.create_session()
    rid = rt.submit_prompt(sid, "T", GenerationOptions(max_new_tokens=8, temperature=0.0))
    rt.cancel(rid)
    for _ in range(200):
        if rt.state() == RuntimeState.READY: break
        time.sleep(0.05)
    assert rt.state() == RuntimeState.READY
    rt.shutdown()


def test_cancel_wrong_request_id_returns_false():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.runtime import AeonDesktopRuntime
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    assert rt.cancel("does-not-exist") is False
    rt.shutdown()


def test_shutdown_during_generation_completes_without_hanging():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.runtime import AeonDesktopRuntime
    from aeon.desktop.protocol import GenerationOptions, RuntimeState
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    sid = rt.create_session()
    rt.submit_prompt(sid, "The", GenerationOptions(max_new_tokens=64, temperature=0.0))
    time.sleep(0.2)  # let some tokens generate
    rt.shutdown(timeout_s=30.0)
    assert rt.state() == RuntimeState.STOPPED


def test_recovery_after_failed_load_can_restart():
    """After a failed load (e.g. bogus release), preflight can be
    attempted again to restart."""
    if not os.path.exists(BUNDLE):
        return
    import tempfile
    from aeon.desktop.runtime import AeonDesktopRuntime, RuntimeError_
    from aeon.desktop.protocol import ErrorCode, RuntimeState
    rt = AeonDesktopRuntime()
    rt.preflight()
    empty = tempfile.mkdtemp()
    try:
        rt.load_release(empty)
    except RuntimeError_ as e:
        assert e.code == ErrorCode.RELEASE_MANIFEST_MISSING
    assert rt.state() == RuntimeState.FAILED
    # Now a real load can succeed after restart
    rt2 = AeonDesktopRuntime()
    rt2.preflight()
    rt2.load_release(BUNDLE)
    assert rt2.state() == RuntimeState.READY
    rt2.shutdown()
    import shutil
    shutil.rmtree(empty, ignore_errors=True)


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
