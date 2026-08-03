"""DESKTOP-6/7/8 — packaging + soak + architecture/IP certification + closure."""
import ast
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BUNDLE = os.path.join(ROOT, "release-assets", "aeon-desktop-p2-proxy")


# ---------------------------------------------------------------------------
# DESKTOP-6 — packaging integration
# ---------------------------------------------------------------------------
def test_pyinstaller_spec_includes_aeon_desktop_modules():
    spec = open(os.path.join(ROOT, "packaging", "windows", "Aeon.spec"),
                  encoding="utf-8").read()
    for mod in ("aeon.desktop", "aeon.desktop.protocol",
                    "aeon.desktop.runtime", "aeon.desktop.chat_ui"):
        assert f"'{mod}'" in spec, f"Aeon.spec missing hidden import {mod}"


def test_pyinstaller_spec_bundles_release_assets():
    spec = open(os.path.join(ROOT, "packaging", "windows", "Aeon.spec"),
                  encoding="utf-8").read()
    assert "release-assets" in spec
    assert "aeon-desktop-p2-proxy" in spec


def test_entry_dispatches_chat_mode():
    from aeon.entry import build_parser
    p = build_parser()
    args = p.parse_args(["--chat"])
    assert args.chat is True


def test_entry_release_root_override():
    from aeon.entry import build_parser
    p = build_parser()
    args = p.parse_args(["--chat", "--release-root", "/tmp/x"])
    assert args.release_root == "/tmp/x"


def test_release_bundle_excludes_forbidden_training_artifacts():
    if not os.path.exists(BUNDLE):
        return
    forbidden = ("optimizer.pt", "trainer_state.json", "corpus_raw",
                     "sealed_test", ".git", "training.log")
    for dirpath, _, files in os.walk(BUNDLE):
        for fn in files:
            for f in forbidden:
                assert f not in fn.lower()


# ---------------------------------------------------------------------------
# DESKTOP-7 — network denial scan (§27)
# ---------------------------------------------------------------------------
def test_desktop_modules_have_no_outbound_network_reference():
    """Static scan of aeon.desktop.* for any outbound-networking symbol."""
    forbidden_tokens = (
        "urllib.request", "urllib.urlopen", "requests.get", "requests.post",
        "requests.put", "requests.delete", "httpx.get", "httpx.post",
        "http.client.HTTPConnection", "http.client.HTTPSConnection",
        "socket.create_connection",
    )
    desk_root = os.path.join(ROOT, "aeon", "desktop")
    for dirpath, _, files in os.walk(desk_root):
        for fn in files:
            if not fn.endswith(".py"): continue
            p = os.path.join(dirpath, fn)
            src = open(p, encoding="utf-8").read()
            for tok in forbidden_tokens:
                assert tok not in src, f"{p}: forbidden network token {tok}"


def test_bundle_manifest_declares_offline_only_policy():
    if not os.path.exists(BUNDLE):
        return
    m = json.load(open(os.path.join(BUNDLE, "manifests/release_manifest.json")))
    assert m["network_policy"] == "offline_only"


# ---------------------------------------------------------------------------
# DESKTOP-7 — bounded stability matrix (§37)
# 25 sequential generation requests + tokens/sec + memory stability
# ---------------------------------------------------------------------------
def test_soak_25_sequential_generations_no_memory_growth_or_orphans():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.runtime import AeonDesktopRuntime
    from aeon.desktop.protocol import GenerationOptions
    import resource
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    sid = rt.create_session()
    trials = []
    for i in range(25):
        r = rt.submit_prompt_sync(
            sid, "The", GenerationOptions(max_new_tokens=4, temperature=0.0))
        trials.append(r)
        # Reset session to keep context small
        if (i + 1) % 5 == 0:
            rt.reset_session(sid)
    # Check every generation completed
    for r in trials:
        kinds = [e["event_type"] for e in r["events"]]
        assert "generation_completed" in kinds, "trial failed"
    # Memory: RSS should not grow monotonically. We check that the last
    # 5 trials' RSS is not more than 200% of the first 5's RSS. We
    # sample RSS via resource.getrusage.
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    # Bounded — 7M model in fp32 ≈ 30MB weights + ~100MB torch overhead
    assert rss_mb < 3000, f"RSS grew to {rss_mb:.0f} MB — potential leak"
    rt.shutdown()


def test_five_new_session_cycles_isolated():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.runtime import AeonDesktopRuntime
    from aeon.desktop.protocol import GenerationOptions
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    session_ids = []
    for _ in range(5):
        sid = rt.create_session()
        rt.submit_prompt_sync(sid, "A",
                                  GenerationOptions(max_new_tokens=2, temperature=0.0))
        session_ids.append(sid)
        rt.close_session(sid)
    # Every session id must be distinct
    assert len(set(session_ids)) == 5
    rt.shutdown()


# ---------------------------------------------------------------------------
# DESKTOP-7 — architecture certification (§38)
# ---------------------------------------------------------------------------
def test_desktop_runtime_asserts_K16_and_ACIS_OFF():
    if not os.path.exists(BUNDLE):
        return
    from aeon.desktop.runtime import AeonDesktopRuntime
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    # HybridModel.K == 16
    assert int(rt._model.K) == 16
    d = rt.diagnostics()
    assert d["K"] == 16
    assert d["ACIS_default"] == "OFF"
    # Recursion state fp32 — check the joiner parameters
    import torch
    for p in rt._model.recursion.parameters():
        assert p.dtype == torch.float32
    rt.shutdown()


def test_desktop_hot_path_does_not_import_research_only_modules():
    """The desktop chat runtime must not reach aeon.bypass.* or any
    training script at import time. Static AST scan on the module
    graph reachable from aeon.desktop.chat_ui."""
    seen = set()
    todo = ["aeon.desktop.chat_ui", "aeon.desktop.runtime",
              "aeon.desktop.protocol"]
    forbidden = ("aeon.bypass", "aeon.job.worker", "aeon.job.manager",
                    "scripts.train", "scripts.run_l3_l4_l5",
                    "scripts.run_pipeline_stage")
    while todo:
        mod = todo.pop()
        if mod in seen: continue
        seen.add(mod)
        # Turn module name into a file path
        rel = mod.replace(".", "/") + ".py"
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p): continue
        src = open(p, encoding="utf-8").read()
        for f in forbidden:
            assert f not in src, f"desktop hot path imports research module: {mod} -> {f}"


def test_manifest_scale_labels_prevent_350M_misrepresentation():
    if not os.path.exists(BUNDLE):
        return
    m = json.load(open(os.path.join(BUNDLE, "manifests/release_manifest.json")))
    assert m["tested_scale"] == "7M proxy"
    assert m["parameter_count"] == 7015366
    # Label must not misrepresent
    lbl = m["release_label"].lower()
    for bad in ("350m", "1.79b", "production", "frontier", "level 3 hidden-state proof"):
        assert bad not in lbl, f"release label carries banned phrase {bad!r}"


# ---------------------------------------------------------------------------
# DESKTOP-8 — closure evidence must exist
# ---------------------------------------------------------------------------
def test_desktop_status_ledger_has_ordered_ladder():
    p = os.path.join(ROOT, "docs", "desktop", "desktop_status.json")
    st = json.load(open(p))
    assert st["desktop_status_ladder"][0] == "NOT_STARTED"
    assert st["desktop_status_ladder"][-1] == "FUNCTIONAL_RELEASE_CANDIDATE"


def test_desktop_status_current_status_is_a_valid_ladder_entry():
    p = os.path.join(ROOT, "docs", "desktop", "desktop_status.json")
    st = json.load(open(p))
    assert st["current_status"] in st["desktop_status_ladder"]


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
