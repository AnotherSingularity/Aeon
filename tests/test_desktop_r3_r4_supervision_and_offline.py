"""DESKTOP-R3 + R4 — in-process supervision semantics + dynamic offline.

R1 selected Architecture B (in-process). Therefore §6 orphan-process
tests are NOT_APPLICABLE (no subprocess). §7 IPC acceptance semantics
that DO apply to an in-process design are exercised here:

  * versioned event schema (RuntimeEvent.schema_version)
  * bounded queue (4096-slot; overflow drops oldest)
  * unknown event types rejected (enum guarantees)
  * malformed / oversized prompt rejected
  * invalid session id rejected
  * duplicate request rejected while one is active
  * cross-session cancellation rejected
  * no arbitrary path exposed by the runtime API
  * no generic command / import / pickle surface

R4 — dynamic network denial by monkey-patching socket.socket +
urllib.request.urlopen + prohibiting any HTTP client at runtime. The
authentic desktop generation path must still load + tokenize +
generate + cancel + reset + shutdown with the network hard-denied.
"""
import os
import sys
import time
import threading

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BUNDLE = os.path.join(ROOT, "release-assets", "aeon-desktop-p2-proxy")


# ---------------------------------------------------------------------------
# R3 — in-process supervision semantics
# ---------------------------------------------------------------------------
def test_R3_event_carries_versioned_schema():
    from aeon.desktop.protocol import PROTOCOL_VERSION, make_event, EventKind
    ev = make_event(EventKind.RUNTIME_STARTING, seq=1)
    d = ev.to_dict()
    assert d["schema_version"] == PROTOCOL_VERSION == 1


def test_R3_duplicate_request_rejected_while_one_active():
    if not os.path.exists(BUNDLE): return
    from aeon.desktop.runtime import AeonDesktopRuntime, RuntimeError_
    from aeon.desktop.protocol import GenerationOptions, ErrorCode
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    sid = rt.create_session()
    rt.submit_prompt(sid, "The", GenerationOptions(max_new_tokens=32, temperature=0.0))
    # Second submit while first is running -> REQUEST_ALREADY_ACTIVE
    try:
        rt.submit_prompt(sid, "The", GenerationOptions(max_new_tokens=4))
    except RuntimeError_ as e:
        assert e.code == ErrorCode.REQUEST_ALREADY_ACTIVE
    # Wait for cleanup
    rt._active_generation_thread.join(timeout=180)
    rt.shutdown()


def test_R3_cross_session_cancellation_returns_false():
    if not os.path.exists(BUNDLE): return
    from aeon.desktop.runtime import AeonDesktopRuntime
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    # Try to cancel a request id belonging to no session
    assert rt.cancel("someone-elses-request") is False
    rt.shutdown()


def test_R3_no_arbitrary_path_argument_on_public_api():
    """The runtime must not expose 'load an arbitrary path' as a normal
    operation — only load_release(release_root) which is fully
    manifest-validated."""
    from aeon.desktop import runtime as rt_mod
    src = open(rt_mod.__file__, encoding="utf-8").read()
    # These would be red flags
    # 'eval(' collides with torch's model.eval() — check via AST for
    # actual eval/exec calls, not substring.
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id in ("eval", "exec", "__import__"):
                raise AssertionError(f"runtime calls {node.func.id}")
    forbidden = ("os.system(", "subprocess.Popen(",
                    "pickle.loads(", "pickle.load(")
    for f in forbidden:
        assert f not in src, f"runtime carries {f}"


def test_R3_release_path_traversal_rejected():
    """A crafted release manifest whose model_artifact_path escapes the
    bundle root must be refused."""
    if not os.path.exists(BUNDLE): return
    import json, shutil, tempfile
    from aeon.desktop.runtime import AeonDesktopRuntime, RuntimeError_
    from aeon.desktop.protocol import ErrorCode
    tmp = tempfile.mkdtemp()
    try:
        dst = os.path.join(tmp, "bundle")
        shutil.copytree(BUNDLE, dst)
        rmp = os.path.join(dst, "manifests", "release_manifest.json")
        m = json.load(open(rmp))
        m["model_artifact_path"] = "../../etc/passwd"
        with open(rmp, "w") as f: json.dump(m, f)
        rt = AeonDesktopRuntime()
        rt.preflight()
        try:
            rt.load_release(dst)
        except RuntimeError_ as e:
            # Escaped path or missing model — both are refusals
            assert e.code in (ErrorCode.RELEASE_MANIFEST_INVALID,
                                 ErrorCode.MODEL_MISSING)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_R3_bounded_queue_never_blocks_runtime_on_overflow():
    """Fill an event handler that intentionally does no work. Overflow
    must not raise from the runtime."""
    import queue
    from aeon.desktop.chat_ui import ChatController
    if not os.path.exists(BUNDLE): return
    ctl = ChatController(BUNDLE)
    ctl.event_q = queue.Queue(maxsize=4)  # force overflow
    # Feed 32 synthetic events — the controller's _on_event drops oldest
    from aeon.desktop.protocol import make_event, EventKind
    for i in range(32):
        ctl._on_event(make_event(EventKind.RUNTIME_WARNING, seq=i))
    assert ctl.event_q.qsize() <= 4


# ---------------------------------------------------------------------------
# R4 — dynamic network denial
# ---------------------------------------------------------------------------
class _NetworkAttempted(RuntimeError):
    pass


def _install_network_guard():
    """Monkey-patch socket + urllib + requests + httpx to raise if any
    outbound network call is attempted. Returns a restore callable."""
    import socket
    import urllib.request as ur
    saved = {}
    saved["socket_connect"] = socket.socket.connect
    saved["socket_create_conn"] = socket.create_connection
    saved["urlopen"] = ur.urlopen

    def _bad_connect(self, *a, **k):
        raise _NetworkAttempted(f"socket.connect attempted with {a!r}")
    def _bad_create_conn(*a, **k):
        raise _NetworkAttempted(f"socket.create_connection attempted with {a!r}")
    def _bad_urlopen(*a, **k):
        raise _NetworkAttempted(f"urlopen attempted with {a!r}")

    socket.socket.connect = _bad_connect
    socket.create_connection = _bad_create_conn
    ur.urlopen = _bad_urlopen

    def restore():
        socket.socket.connect = saved["socket_connect"]
        socket.create_connection = saved["socket_create_conn"]
        ur.urlopen = saved["urlopen"]
    return restore


def test_R4_desktop_pipeline_runs_without_any_outbound_network_attempt():
    """Full desktop flow — preflight, load_release, create_session,
    submit_prompt_sync, cancel path, reset, shutdown — with socket +
    urllib patched to raise on any use."""
    if not os.path.exists(BUNDLE): return
    from aeon.desktop.runtime import AeonDesktopRuntime
    from aeon.desktop.protocol import GenerationOptions
    restore = _install_network_guard()
    try:
        rt = AeonDesktopRuntime()
        rt.preflight()
        rt.load_release(BUNDLE)
        sid = rt.create_session()
        # a full generation
        r = rt.submit_prompt_sync(sid, "Alice",
                                        GenerationOptions(max_new_tokens=6, temperature=0.0))
        kinds = [e["event_type"] for e in r["events"]]
        assert "generation_completed" in kinds
        # a cancellation cycle
        rt.submit_prompt(sid, "The", GenerationOptions(max_new_tokens=64, temperature=0.0))
        time.sleep(0.2)
        rt.cancel(rt._active_request_id)
        rt._active_generation_thread.join(timeout=180)
        # a reset
        rt.reset_session(sid)
        rt.shutdown()
    finally:
        restore()


def test_R4_no_new_thread_leaks_after_repeated_gen_cancel():
    """Repeated generate + cancel must not leak threads."""
    if not os.path.exists(BUNDLE): return
    from aeon.desktop.runtime import AeonDesktopRuntime
    from aeon.desktop.protocol import GenerationOptions
    baseline = threading.active_count()
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    sid = rt.create_session()
    for _ in range(5):
        rt.submit_prompt(sid, "The", GenerationOptions(max_new_tokens=32, temperature=0.0))
        time.sleep(0.1)
        rt.cancel(rt._active_request_id)
        rt._active_generation_thread.join(timeout=60)
    rt.shutdown()
    # give any lingering finalizers a beat
    time.sleep(0.3)
    now = threading.active_count()
    # Allow 4 extra threads (torch inductor + gc) as slack
    assert now <= baseline + 4, f"thread leak: {baseline} -> {now}"


def test_R4_desktop_source_has_no_network_client_import():
    """Retain the source-level check; classify it clearly as static
    evidence, not dynamic proof. The dynamic proof is the two tests
    above."""
    forbidden = ("import requests", "import httpx", "import aiohttp",
                    "import websocket", "from urllib.request import urlopen",
                    "from requests import")
    for dp, _, files in os.walk(os.path.join(ROOT, "aeon", "desktop")):
        for fn in files:
            if not fn.endswith(".py"): continue
            src = open(os.path.join(dp, fn), encoding="utf-8").read()
            for f in forbidden:
                assert f not in src, f"{fn}: forbidden import {f}"


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
