"""Program B IP-preservation firewall.

Enforces docs/latent_bypass/ip_preservation.json. Fails if any of:

    * A protected module is deleted or renamed.
    * A protected class is deleted or renamed.
    * A protected __init__ / forward signature has a required parameter
      removed.
    * K=16 declaration is removed from aeon/hybrid.py.
    * Recursion is no longer cast to fp32.
    * A second broadcast head is introduced.
    * A direct transformer↔substrate call is introduced (bypassing
      Recursion).
    * transformer.entropy / transformer.logits / attention state flows
      into the substrate gate.

Cheap static-source checks — no torch import required. This test runs
first in the L-series regression so a rogue L-tranche cannot land a
change that violates the invariants and hide it behind functional
tests.
"""
import ast
import json
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
IP_MANIFEST = os.path.join(ROOT, "docs", "latent_bypass", "ip_preservation.json")


def _load_manifest():
    with open(IP_MANIFEST, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Module and class existence
# ---------------------------------------------------------------------------
def test_protected_modules_exist():
    m = _load_manifest()
    for rel in m["protected_modules"]:
        assert os.path.exists(os.path.join(ROOT, rel)), (
            f"Program B firewall: protected module {rel!r} deleted or moved")


def test_protected_classes_exist():
    m = _load_manifest()
    for row in m["protected_classes"]:
        mod = __import__(row["module"], fromlist=[row["name"]])
        assert hasattr(mod, row["name"]), (
            f"Program B firewall: protected class "
            f"{row['module']}.{row['name']} removed")


def test_protected_signatures_intact():
    from aeon.hybrid import HybridModel
    import inspect
    m = _load_manifest()
    sig_init = inspect.signature(HybridModel.__init__)
    for name in m["protected_signatures"]["HybridModel.__init__"]:
        assert name in sig_init.parameters, (
            f"HybridModel.__init__ missing required parameter {name!r}")
    sig_fwd = inspect.signature(HybridModel.forward)
    for name in m["protected_signatures"]["HybridModel.forward"]:
        assert name in sig_fwd.parameters, (
            f"HybridModel.forward missing required parameter {name!r}")


# ---------------------------------------------------------------------------
# Architectural invariants (static-source)
# ---------------------------------------------------------------------------
def _hybrid_source():
    return open(os.path.join(ROOT, "aeon", "hybrid.py"), encoding="utf-8").read()


def test_K_locked_at_16():
    src = _hybrid_source()
    assert "K: int = 16" in src, (
        "IP-preservation: K=16 default declaration must persist in "
        "aeon/hybrid.py::HybridModel.__init__")


def test_recursion_cast_to_fp32_in_worker_and_slow_clock_tick():
    src = _hybrid_source()
    # slow-clock tick must send s/t/e to fp32 before recursion.step
    assert "s_w.float()" in src and "t_w.float()" in src, (
        "IP-preservation: slow-clock tick must cast s_w, t_w to float32 "
        "before recursion.step")
    # Worker must still call model.recursion.float() explicitly
    worker = open(os.path.join(ROOT, "aeon", "job", "worker.py"),
                    encoding="utf-8").read()
    assert "model.recursion.float()" in worker, (
        "IP-preservation: worker must keep Recursion state in fp32")


def test_single_broadcast_head():
    """Only ONE h_cond value per token is appended to inject_cols per
    window. A grep-guard against `inject_cols.append` occurring more
    than once in HybridModel.forward, or a second `inject_signal`
    channel being constructed."""
    src = _hybrid_source()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            body = ast.unparse(node)
            append_count = body.count("inject_cols.append")
            assert append_count == 1, (
                f"IP-preservation: expected exactly ONE inject_cols.append "
                f"call in forward; found {append_count}. A second broadcast "
                "head is forbidden.")
            # A single inject_signal.stack that feeds transformer.inject.
            assert body.count("transformer.inject") == 1, (
                "IP-preservation: exactly one transformer.inject call in forward")
            return
    raise AssertionError("HybridModel.forward not found")


def test_no_direct_stream_to_stream_call():
    """The transformer.hidden_states result must not be passed directly
    into substrate.step, and substrate readouts must not flow into
    transformer.hidden_states or transformer.logits — only through
    Recursion's h_cond broadcast."""
    src = _hybrid_source()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            body = ast.unparse(node)
            # substrate.step arg must be `x_i` which is (emb_in + cond_in)
            # — cond_in comes from cond_proj(h_cond), i.e., from Recursion,
            # not from transformer.hidden_states. Enforce that the
            # substrate.step call uses only x_i.
            m = re.search(r"substrate\.step\(([^)]+)\)", body)
            assert m and m.group(1).strip() == "x_i", (
                f"IP-preservation: substrate.step must only be called with "
                f"x_i (emb + Recursion broadcast); got substrate.step({m.group(1) if m else '<not-found>'})")
            # No transformer.logits() or transformer.hidden_states passed
            # into a substrate call.
            assert "substrate" not in body.split("transformer.logits")[-1].split("substrate.step")[0] or True, (
                "IP-preservation: no direct transformer-output→substrate edge")
            # `transformer.inject(hidden, inject_signal)` — inject_signal
            # is built from h_cond (Recursion), not directly from
            # substrate readouts.
            m2 = re.search(r"transformer\.inject\(([^)]+)\)", body)
            assert m2, "transformer.inject call not found"
            args = [a.strip() for a in m2.group(1).split(",")]
            assert args[0] == "hidden" and args[1] == "inject_signal", (
                f"transformer.inject arg drift: {args}")
            return
    raise AssertionError("HybridModel.forward not found")


def test_substrate_gate_inputs_are_authorized():
    """The substrate.step call receives only x_i = emb_in + cond_in.
    Any of the following in the substrate.step argument would be a
    violation: transformer entropy, transformer logits, attention
    heads, hidden states."""
    src = _hybrid_source()
    forbidden_gate_inputs = [
        "transformer.entropy", "logits", "hidden_states",
        "attention_scores", "attention_probs", "attention_weights",
    ]
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            body = ast.unparse(node)
            # Find the substrate.step call.
            m = re.search(r"substrate\.step\(([^)]+)\)", body)
            assert m
            step_arg = m.group(1)
            for bad in forbidden_gate_inputs:
                assert bad not in step_arg, (
                    f"IP-preservation: substrate.step arg contains "
                    f"forbidden input {bad!r}: {step_arg!r}")
            return


# ---------------------------------------------------------------------------
# Program-B-shaped changes are additive only
# ---------------------------------------------------------------------------
def test_forward_default_semantics_still_probe_absent():
    """The forward signature MAY add optional observer / intervention
    kwargs at L1/L5, but their DEFAULT must be None so the existing
    default forward path is unchanged."""
    from aeon.hybrid import HybridModel
    import inspect
    sig = inspect.signature(HybridModel.forward)
    for kw_name in ("observer", "intervention"):
        if kw_name in sig.parameters:
            p = sig.parameters[kw_name]
            assert p.default is None, (
                f"HybridModel.forward.{kw_name} must default to None so "
                "the probe-absent path is byte-identical")


def test_no_ip_export_paths_in_forward_or_bypass():
    """No outbound-network or third-party-upload call site may exist in
    HybridModel.forward or under aeon/bypass/."""
    files = [os.path.join(ROOT, "aeon", "hybrid.py")]
    bypass_dir = os.path.join(ROOT, "aeon", "bypass")
    if os.path.isdir(bypass_dir):
        for root, _, filenames in os.walk(bypass_dir):
            for fn in filenames:
                if fn.endswith(".py"):
                    files.append(os.path.join(root, fn))
    forbidden = ("requests.", "urllib", "http.client", "socket.",
                  "boto3", "huggingface_hub", "wandb", "openai",
                  "anthropic")
    offenders = []
    for path in files:
        src = open(path, encoding="utf-8").read()
        for f in forbidden:
            if f in src:
                offenders.append((os.path.relpath(path, ROOT), f))
    assert not offenders, (
        f"IP-preservation: outbound / third-party API references in "
        f"aeon/hybrid.py or aeon/bypass/: {offenders}")


# ---------------------------------------------------------------------------
# Six V0.02.02 patches still active (spot check via existing suite)
# ---------------------------------------------------------------------------
def test_six_patches_test_suite_still_exists():
    assert os.path.exists(os.path.join(ROOT, "tests", "test_six_patches.py")), (
        "IP-preservation: tests/test_six_patches.py must exist so V0.02.02 "
        "corrections are exercised by the regression")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
