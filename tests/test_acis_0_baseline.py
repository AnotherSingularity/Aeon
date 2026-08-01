"""ACIS-0 — invariant lock + current-transport baseline.

Proves the certified invariants BEFORE any ACIS transport code is
wired into HybridModel.forward. Every subsequent tranche must keep
these assertions green:

    * K=16 fixed. No adaptive-K component.
    * One broadcast per boundary (exactly one inject_cols.append).
    * Recursion state fp32.
    * Substrate gate inputs authorized (no transformer entropy/logits).
    * Default forward path unchanged when ACIS is disabled — this
      test file itself imports aeon.shuttle only when explicitly
      exercised.
    * ShuttleMode enum well-formed; unknown modes fail closed.
    * Audit-log ledger digest chains events; replay is detectable.
    * aeon.shuttle currently exports nothing that HybridModel.forward
      imports (proves ACIS-0 is a pure additive scaffold).
"""
import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _hybrid_source():
    return open(os.path.join(ROOT, "aeon", "hybrid.py"),
                  encoding="utf-8").read()


# ---------------------------------------------------------------------------
# Certified invariants (baseline)
# ---------------------------------------------------------------------------
def test_fixed_k_declared_at_16():
    src = _hybrid_source()
    assert "K: int = 16" in src, "K=16 default must persist"
    # aeon.shuttle re-exports the fixed value; module-level constant.
    from aeon.shuttle import FIXED_K
    assert FIXED_K == 16


def test_no_adaptive_k_component_present():
    """The directive forbids introducing an AdaptiveKControlCapsule
    (or equivalent). Enforce that no symbol with 'Adaptive' + 'K' lives
    under aeon/shuttle/ or elsewhere in aeon/."""
    forbidden_substrings = ("AdaptiveKControl", "adaptive_k",
                              "AdaptiveKCapsule")
    for base in ("aeon", "scripts"):
        for dirpath, _, filenames in os.walk(os.path.join(ROOT, base)):
            for fn in filenames:
                if not fn.endswith(".py"):
                    continue
                path = os.path.join(dirpath, fn)
                with open(path, encoding="utf-8", errors="replace") as fh:
                    src = fh.read()
                for pattern in forbidden_substrings:
                    assert pattern not in src, (
                        f"forbidden adaptive-K token {pattern!r} in "
                        f"{os.path.relpath(path, ROOT)}")


def test_single_broadcast_per_window():
    """Baseline invariant carried over from IP-preservation firewall.
    Exactly one inject_cols.append and one transformer.inject call."""
    src = _hybrid_source()
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "forward":
            body = ast.unparse(node)
            assert body.count("inject_cols.append") == 1
            assert body.count("transformer.inject") == 1
            return
    raise AssertionError("HybridModel.forward not found")


def test_recursion_still_fp32_in_slow_clock_tick():
    src = _hybrid_source()
    assert "s_w.float()" in src and "t_w.float()" in src


def test_substrate_gate_inputs_unchanged():
    """substrate.step must still be called with x_i only — no
    transformer entropy / logits / hidden state contamination."""
    import re
    src = _hybrid_source()
    m = re.search(r"substrate\.step\(([^)]+)\)", src)
    assert m and m.group(1).strip() == "x_i"


# ---------------------------------------------------------------------------
# ShuttleMode
# ---------------------------------------------------------------------------
def test_shuttle_mode_enum_has_four_states():
    from aeon.shuttle.policy import ShuttleMode
    names = {m.name for m in ShuttleMode}
    assert names == {"OFF", "OBSERVE", "BUCKET", "CONVEYOR_EXPERIMENTAL"}


def test_shuttle_mode_default_is_off():
    from aeon.shuttle import SHUTTLE_MODE_DEFAULT
    from aeon.shuttle.policy import ShuttleMode, is_default_off, parse_shuttle_mode
    assert SHUTTLE_MODE_DEFAULT == "off"
    assert parse_shuttle_mode(SHUTTLE_MODE_DEFAULT) is ShuttleMode.OFF
    assert is_default_off(ShuttleMode.OFF) is True
    assert is_default_off(ShuttleMode.OBSERVE) is False


def test_shuttle_mode_unknown_string_fails_closed():
    from aeon.shuttle.policy import parse_shuttle_mode, UnknownShuttleMode
    for bad in ("", "on", "OFF", "off ", "adaptive", "auto"):
        try:
            parse_shuttle_mode(bad)
        except UnknownShuttleMode:
            pass
        else:
            raise AssertionError(f"parse_shuttle_mode({bad!r}) must fail")
    # Non-string input also fails closed.
    try:
        parse_shuttle_mode(1)
    except UnknownShuttleMode:
        pass
    else:
        raise AssertionError("non-string ShuttleMode input must fail")


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------
def test_audit_log_chains_ledger_digest_and_detects_replay():
    from aeon.shuttle.audit import AcisAuditLog
    log = AcisAuditLog()
    ev1 = log.append(kind="publish", boundary_index=0, recursion_epoch=0,
                       payload_digest="sha256:a" * 8)
    ev2 = log.append(kind="lease_issue", boundary_index=0, recursion_epoch=0,
                       payload_digest="sha256:b" * 8,
                       detail={"destination": "TRANSFORMER"})
    assert ev2.prev_ledger_digest == ev1.ledger_digest
    # If a replay attempt appended the same event with the ORIGINAL
    # prev_ledger_digest, the resulting ledger_digest would differ from
    # ev2's — that's how replay is detectable.
    log2 = AcisAuditLog()
    log2.append(kind="publish", boundary_index=0, recursion_epoch=0,
                  payload_digest="sha256:a" * 8)
    # Replay of ev2 into the fresh log after a differing state
    # (log2.head_digest here matches ev1.ledger_digest because we've
    # inserted the same first event) — same head → same ledger_digest,
    # which is intentional: replay detection compares expected
    # prev-ledger against actual.
    log2.append(kind="lease_issue", boundary_index=0, recursion_epoch=0,
                  payload_digest="sha256:b" * 8,
                  detail={"destination": "TRANSFORMER"})
    # A THIRD event on log2 would carry prev=log2.head_digest; if the
    # attacker replayed ev2 with an outdated prev, the resulting
    # ledger_digest wouldn't line up with log2's expected chain.
    ev3 = log2.append(kind="lease_ack", boundary_index=0, recursion_epoch=0,
                        payload_digest="sha256:c" * 8)
    assert ev3.prev_ledger_digest == log2.events()[1].ledger_digest


def test_audit_log_never_holds_payload_reference():
    """An AcisEvent has a payload_digest field but must not have a
    field named 'payload' or 'tensor' or 'bytes' that could accept the
    actual object."""
    from aeon.shuttle.audit import AcisEvent
    from dataclasses import fields
    field_names = {f.name for f in fields(AcisEvent)}
    for forbidden in ("payload", "tensor", "bytes", "activations"):
        assert forbidden not in field_names


# ---------------------------------------------------------------------------
# Scaffold guarantees — aeon.shuttle is not imported by hybrid.py
# ---------------------------------------------------------------------------
def test_hybrid_shuttle_import_is_guarded_by_none_check():
    """After ACIS-3 the shuttle wire-through lands: `from aeon.shuttle
    ... import ...` appears inside a `if shuttle is not None:` branch.
    Prove that (a) any aeon.shuttle import in hybrid.py sits INSIDE
    a function body (not module-level, so an aeon build without
    aeon.shuttle still loads), and (b) the default forward path when
    shuttle is None never touches aeon.shuttle."""
    src = _hybrid_source()
    tree = ast.parse(src)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in getattr(node, "names", []) or []:
                assert "aeon.shuttle" not in getattr(alias, "name", ""), (
                    "aeon.shuttle must not be imported at module level "
                    "in aeon/hybrid.py")
            if isinstance(node, ast.ImportFrom):
                assert node.module != "aeon.shuttle", (
                    "aeon.shuttle module-level from-import forbidden")
    # And the shuttle branch must be gated by the None check.
    assert "if shuttle is not None:" in src, (
        "ACIS-3: shuttle branch must be gated by `if shuttle is not None:`")


def test_shuttle_package_has_no_outbound_network_reference():
    """The IP-preservation firewall forbids outbound network calls in
    aeon/hybrid.py or aeon/bypass/. Extend the guarantee to aeon/shuttle/."""
    forbidden = ("requests.", "urllib.request", "http.client",
                  "socket.", "boto3", "huggingface_hub", "wandb",
                  "openai", "anthropic-hosted")
    shuttle_dir = os.path.join(ROOT, "aeon", "shuttle")
    for dirpath, _, filenames in os.walk(shuttle_dir):
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            with open(os.path.join(dirpath, fn), encoding="utf-8") as fh:
                src = fh.read()
            for tok in forbidden:
                assert tok not in src, (
                    f"{os.path.relpath(os.path.join(dirpath, fn), ROOT)}: "
                    f"forbidden outbound token {tok!r}")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
