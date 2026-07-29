"""
F9.1 — Permanent evidence-path canonicalization tests.

Two layers:

  1. Focused unit tests for `aeon/evidence.py::normalize_path_string` covering
     the 16 required cases from the directive.
  2. Repository-wide structured scan over committed JSON evidence bundles that
     asserts none carry POSIX/Windows host-specific paths or the current
     machine's repo/home/tmp prefixes.

Torch-free.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# 16 focused unit tests
# ---------------------------------------------------------------------------
def test_01_posix_repository_path():
    from aeon.evidence import normalize_path_string, DEFAULT_REPO_ROOT
    s = f"loaded {DEFAULT_REPO_ROOT}/runs/aeon/ckpt_1.pt at step 5"
    out = normalize_path_string(s)
    assert "<repo>/runs/aeon/ckpt_1.pt" in out, out
    assert DEFAULT_REPO_ROOT not in out


def test_02_posix_temporary_path():
    from aeon.evidence import normalize_path_string
    s = "wrote /tmp/tmpABC123xyz/fixture.pt at step 1"
    out = normalize_path_string(s)
    assert "<tmp>" in out and "tmpABC123" not in out, out


def test_03_posix_home_directory_path():
    from aeon.evidence import normalize_path_string
    s = "read /home/alice/data/x.txt failed"
    out = normalize_path_string(s, home="/home/alice")
    assert "<home>/data/x.txt" in out, out
    assert "alice" not in out


def test_04_arbitrary_posix_absolute_path():
    from aeon.evidence import normalize_path_string
    s = "denied read: '/etc/passwd' outside allowed roots"
    out = normalize_path_string(s)
    assert "<absolute>/passwd" in out, out
    assert "/etc" not in out


def test_05_windows_repository_path():
    from aeon.evidence import normalize_path_string
    s = r"opened C:\Users\bob\projects\AeonV0.02\runs\ck.pt"
    out = normalize_path_string(s, repo_root=r"C:\Users\bob\projects\AeonV0.02",
                                  home=r"C:\Users\bob")
    # <repo>/runs/ck.pt — forward slashes, no bob, no drive
    assert "<repo>/runs/ck.pt" in out, out
    assert "bob" not in out and "C:" not in out


def test_06_windows_temporary_path():
    from aeon.evidence import normalize_path_string
    s = r"error at C:\Users\bob\AppData\Local\Temp\pytest-42\fixture.pt"
    out = normalize_path_string(s)
    assert "<tmp>/pytest-42/fixture.pt" in out, out
    assert "bob" not in out


def test_07_windows_user_profile_path():
    from aeon.evidence import normalize_path_string
    s = r"loaded C:\Users\alice\Documents\model.pt"
    out = normalize_path_string(s)
    assert "<home>/Documents/model.pt" in out, out
    assert "alice" not in out


def test_08_nested_dict_and_list_values():
    from aeon.evidence import sanitize_evidence, DEFAULT_REPO_ROOT
    obj = {"outer": [{"path": f"{DEFAULT_REPO_ROOT}/x", "tag": "keep"},
                     {"nested": {"y": "/tmp/tmpAB12CD34/z"}}]}
    out = sanitize_evidence(obj)
    # keys untouched (no path shapes); values rewritten
    assert out["outer"][0]["path"] == "<repo>/x"
    assert out["outer"][0]["tag"] == "keep"
    assert out["outer"][1]["nested"]["y"] == "<tmp>/z"


def test_09_exception_messages_with_multiple_paths():
    from aeon.evidence import normalize_path_string, DEFAULT_REPO_ROOT
    s = (f"FileNotFoundError: {DEFAULT_REPO_ROOT}/runs/ck.pt not found; "
         f"fallback checked /tmp/tmpXYZ123abc/ckpt.pt and /home/user/backup.pt")
    out = normalize_path_string(s, home="/home/user")
    # Exception class preserved
    assert out.startswith("FileNotFoundError:")
    # All three paths rewritten
    assert "<repo>/runs/ck.pt" in out
    assert "<tmp>" in out and "tmpXYZ123abc" not in out
    assert "<home>/backup.pt" in out


def test_10_relative_paths_stay_relative():
    from aeon.evidence import normalize_path_string
    for s in ("configs/aeon_350m_primary.yaml", "docs/F1_THREAT_MODEL.md",
              "runs/aeon_e5/instrumented/metrics.jsonl", "aeon/hybrid.py"):
        out = normalize_path_string(s)
        assert out == s, f"relative changed: {s!r} -> {out!r}"


def test_11_urls_unchanged():
    from aeon.evidence import normalize_path_string
    for s in ("https://example.org/a/b/c",
              "http://download.pytorch.org/whl/cu124",
              "git+https://github.com/foo/bar.git@main"):
        out = normalize_path_string(s, repo_root="/home/x/y", home="/home/x")
        assert out == s, f"url changed: {s!r} -> {out!r}"


def test_12_hashes_unchanged():
    from aeon.evidence import normalize_path_string
    # sha1 (40), sha256 (64), sha3-224 (56)
    for h in ("da39a3ee5e6b4b0d3255bfef95601890afd80709",
              "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
              "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f"):
        s = f"digest={h}"
        out = normalize_path_string(s)
        assert h in out, f"hash changed: {out}"


def test_13_slash_containing_non_paths_unchanged():
    from aeon.evidence import normalize_path_string
    for s in ("a/b/c is a relative fragment",
              "ratio 3/4 of frames",
              "date 2025/07/29 recorded",
              "MIT/Apache-2.0 licence"):
        out = normalize_path_string(s)
        assert out == s, f"non-path changed: {s!r} -> {out!r}"


def test_14_evidence_generated_by_f6_after_sanitize():
    """Simulate an F6-shaped raw record; sanitize; assert no leaks."""
    from aeon.evidence import sanitize_evidence, DEFAULT_REPO_ROOT
    raw = {
        "phase": "F6",
        "cases": [{
            "actual_response": (
                f"RuntimePolicyError: denied read: '{DEFAULT_REPO_ROOT}/etc/passwd' "
                f"and /tmp/tmpDEADBEEF/repo/x outside allowed roots"),
            "audit_event_id": "abcdef1234567890",
        }],
    }
    out = sanitize_evidence(raw)
    text = json.dumps(out)
    assert DEFAULT_REPO_ROOT not in text
    assert "tmpDEADBEEF" not in text
    assert "<tmp>" in text or "<absolute>" in text
    # the audit id was hex-like but too short to be a hash — check it wasn't
    # rewritten (it's not a path either)
    assert "abcdef1234567890" in text


def test_15_evidence_generated_by_f8_after_sanitize():
    from aeon.evidence import sanitize_evidence
    raw = {"exercises": [{
        "name": "corrupted_newest_ckpt",
        "checkpoint": "/tmp/tmpQ89xyz/ex_01/ck.pt",
        "audit_log": "/tmp/tmpQ89xyz/ex_01/audit.jsonl",
        "recovery_time_s": 0.012,
    }]}
    out = sanitize_evidence(raw)
    for k in ("checkpoint", "audit_log"):
        v = out["exercises"][0][k]
        assert "tmpQ89xyz" not in v, v
        assert v.startswith("<tmp>"), v
    assert out["exercises"][0]["recovery_time_s"] == 0.012


def test_16_repeated_runs_with_different_tmp_produce_equivalent_sanitized_output():
    from aeon.evidence import sanitize_evidence, canonical_json_bytes
    a = {"case": "x", "ckpt": "/tmp/tmpAAA111zzz/ck.pt", "audit": "/tmp/tmpAAA111zzz/audit.jsonl"}
    b = {"case": "x", "ckpt": "/tmp/tmpBBB222yyy/ck.pt", "audit": "/tmp/tmpBBB222yyy/audit.jsonl"}
    ha = canonical_json_bytes(sanitize_evidence(a))
    hb = canonical_json_bytes(sanitize_evidence(b))
    assert ha == hb, (ha, hb)


# ---------------------------------------------------------------------------
# Extra tests — determinism + repo-wide scan
# ---------------------------------------------------------------------------
def test_write_evidence_is_deterministic_across_calls():
    from aeon.evidence import write_evidence
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ev.json")
        sha_a = write_evidence(p, {"z": 1, "a": [3, 2, 1], "b": "/home/x/y"},
                                home="/home/x")
        sha_b = write_evidence(p, {"a": [3, 2, 1], "z": 1, "b": "/home/x/y"},
                                home="/home/x")
        assert sha_a == sha_b


def test_repo_wide_json_evidence_scan_is_clean():
    """Structured JSON-field inspection over every committed evidence JSON.
    Uses aeon.evidence.scan_json_for_host_paths — not a naive text grep — so
    documentation Markdown that intentionally discusses prohibited patterns is
    not flagged."""
    import glob
    from aeon.evidence import scan_json_for_host_paths
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # Committed evidence bundles + policy JSONs
    targets = []
    for name in ("evidence", "results", "accounting", "baseline", "policy",
                  "manifest", "topology", "provenance", "audit", "registry",
                  "schema", "tests"):
        targets += glob.glob(os.path.join(root, "docs", f"*{name}*.json"))
    # Deduplicate
    targets = sorted(set(targets))
    all_offenders = {}
    for path in targets:
        off = scan_json_for_host_paths(path)
        if off:
            all_offenders[os.path.relpath(path, root)] = off
    assert not all_offenders, "committed evidence contains host paths:\n" + \
        "\n".join(f"  {f}:\n    " + "\n    ".join(v) for f, v in all_offenders.items())


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
