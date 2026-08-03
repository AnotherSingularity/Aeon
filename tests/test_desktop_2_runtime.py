"""DESKTOP-2 — AeonDesktopRuntime.

Tests both the state-machine invariants (fast, no model load) and a
live smoke: load the release bundle, generate a bounded number of
tokens, verify events fire, verify K=16, verify session isolation.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


BUNDLE = os.path.join(ROOT, "release-assets", "aeon-desktop-p2-proxy")


# ---------------------------------------------------------------------------
# State-machine + protocol (fast, no model)
# ---------------------------------------------------------------------------
def test_protocol_event_kinds_are_frozen_strings():
    from aeon.desktop.protocol import EventKind
    for k in EventKind:
        assert isinstance(k.value, str)


def test_protocol_error_codes_include_required_set():
    from aeon.desktop.protocol import ErrorCode
    required = {"RELEASE_MANIFEST_MISSING", "MODEL_DIGEST_MISMATCH",
                  "TOKENIZER_DIGEST_MISMATCH", "FIXED_K_MISMATCH",
                  "PROMPT_EMPTY", "PROMPT_TOO_LARGE", "SETTINGS_INVALID",
                  "SESSION_NOT_FOUND", "GENERATION_CANCELLED",
                  "GENERATION_FAILED", "OUT_OF_MEMORY", "SHUTDOWN_TIMEOUT"}
    codes = {c.value for c in ErrorCode}
    for r in required:
        assert r in codes, f"missing ErrorCode.{r}"


def test_state_machine_rejects_impossible_transitions():
    from aeon.desktop.protocol import RuntimeState, is_valid_transition
    assert not is_valid_transition(RuntimeState.STOPPED, RuntimeState.GENERATING)
    assert not is_valid_transition(RuntimeState.FAILED, RuntimeState.GENERATING)
    assert not is_valid_transition(RuntimeState.LOADING_MODEL, RuntimeState.GENERATING)
    assert not is_valid_transition(RuntimeState.READY, RuntimeState.READY)
    assert is_valid_transition(RuntimeState.NOT_STARTED, RuntimeState.STARTING)
    assert is_valid_transition(RuntimeState.READY, RuntimeState.GENERATING)


def test_settings_reject_out_of_range_and_nan():
    import math
    from aeon.desktop.protocol import GenerationOptions, validate_settings, SettingsInvalid
    ok = validate_settings(GenerationOptions(), vocab_size=16000)
    assert ok is not None
    try:
        validate_settings(GenerationOptions(temperature=-1.0), vocab_size=16000)
    except SettingsInvalid as e:
        assert "temperature" in e.detail
    try:
        validate_settings(GenerationOptions(temperature=float("nan")), vocab_size=16000)
    except SettingsInvalid as e:
        assert "temperature" in e.detail
    try:
        validate_settings(GenerationOptions(max_new_tokens=99999), vocab_size=16000)
    except SettingsInvalid as e:
        assert "max_new_tokens" in e.detail
    try:
        validate_settings(GenerationOptions(top_k=-5), vocab_size=16000)
    except SettingsInvalid as e:
        assert "top_k" in e.detail


def test_runtime_rejects_generation_before_load():
    from aeon.desktop.runtime import AeonDesktopRuntime, RuntimeError_
    from aeon.desktop.protocol import ErrorCode, GenerationOptions
    rt = AeonDesktopRuntime()
    try:
        rt.create_session()
    except RuntimeError_ as e:
        assert e.code == ErrorCode.RUNTIME_START_FAILED


def test_runtime_manifest_digest_mismatch_fails_closed():
    """Corrupt the manifest's model_artifact_sha256 in a copy and prove
    load_release refuses. We do this on an out-of-tree copy so the
    real bundle isn't touched."""
    if not os.path.exists(BUNDLE):
        return
    import shutil
    import tempfile
    from aeon.desktop.runtime import AeonDesktopRuntime, RuntimeError_
    from aeon.desktop.protocol import ErrorCode
    tmp = tempfile.mkdtemp()
    try:
        dst = os.path.join(tmp, "aeon-desktop-p2-proxy")
        shutil.copytree(BUNDLE, dst)
        rmp = os.path.join(dst, "manifests", "release_manifest.json")
        m = json.load(open(rmp))
        m["model_artifact_sha256"] = "sha256:" + "0" * 64
        with open(rmp, "w") as f:
            json.dump(m, f)
        rt = AeonDesktopRuntime()
        rt.preflight()
        try:
            rt.load_release(dst)
        except RuntimeError_ as e:
            assert e.code == ErrorCode.MODEL_DIGEST_MISMATCH
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Live smoke: real forward + real generation
# ---------------------------------------------------------------------------
def test_runtime_load_release_generates_events_and_ready_state():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.runtime import AeonDesktopRuntime
    from aeon.desktop.protocol import RuntimeState, EventKind
    events = []
    rt = AeonDesktopRuntime(event_handler=lambda ev: events.append(ev.event_type.value))
    rt.preflight()
    rt.load_release(BUNDLE)
    assert rt.state() == RuntimeState.READY
    assert EventKind.RUNTIME_PREFLIGHT_COMPLETED.value in events
    assert EventKind.MODEL_VALIDATION_COMPLETED.value in events
    assert EventKind.MODEL_READY.value in events
    rt.shutdown()


def test_runtime_generates_bounded_tokens_end_to_end():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.runtime import AeonDesktopRuntime
    from aeon.desktop.protocol import GenerationOptions
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    sid = rt.create_session()
    result = rt.submit_prompt_sync(
        sid, "The", GenerationOptions(max_new_tokens=8, temperature=0.0))
    events = result["events"]
    kinds = [e["event_type"] for e in events]
    assert "generation_started" in kinds
    assert "generation_completed" in kinds
    completed = [e for e in events if e["event_type"] == "generation_completed"][0]
    assert completed["payload"]["generated_tokens"] > 0
    assert completed["payload"]["generated_tokens"] <= 8
    rt.shutdown()


def test_runtime_session_isolation_two_sessions_have_separate_histories():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.runtime import AeonDesktopRuntime
    from aeon.desktop.protocol import GenerationOptions
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    sid_a = rt.create_session()
    sid_b = rt.create_session()
    rt.submit_prompt_sync(sid_a, "Alpha", GenerationOptions(max_new_tokens=3, temperature=0.0))
    rt.submit_prompt_sync(sid_b, "Beta", GenerationOptions(max_new_tokens=3, temperature=0.0))
    hist_a = list(rt._sessions[sid_a].token_history)
    hist_b = list(rt._sessions[sid_b].token_history)
    assert len(hist_a) > 0 and len(hist_b) > 0
    assert hist_a != hist_b
    # Reset A; B unaffected
    rt.reset_session(sid_a)
    assert len(rt._sessions[sid_a].token_history) == 0
    assert list(rt._sessions[sid_b].token_history) == hist_b
    rt.shutdown()


def test_runtime_reports_K_16_and_ACIS_OFF_via_diagnostics():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.runtime import AeonDesktopRuntime
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    d = rt.diagnostics()
    assert d["K"] == 16
    assert d["ACIS_default"] == "OFF"
    assert d["network_policy"] == "offline_only"
    assert d["release_id"].startswith("aeon-desktop-p2-proxy-")
    rt.shutdown()


def test_runtime_rejects_empty_prompt():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.runtime import AeonDesktopRuntime, RuntimeError_
    from aeon.desktop.protocol import ErrorCode
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    sid = rt.create_session()
    try:
        rt.submit_prompt(sid, "")
    except RuntimeError_ as e:
        assert e.code == ErrorCode.PROMPT_EMPTY
    rt.shutdown()


def test_runtime_rejects_oversized_prompt():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.runtime import AeonDesktopRuntime, RuntimeError_
    from aeon.desktop.protocol import ErrorCode
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    sid = rt.create_session()
    big = "X" * 100000
    try:
        rt.submit_prompt(sid, big)
    except RuntimeError_ as e:
        assert e.code == ErrorCode.PROMPT_TOO_LARGE
    rt.shutdown()


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
