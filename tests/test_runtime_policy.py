"""
F4 — Runtime containment tests.

Verifies:
  §F4.1 execution identity attestations are present in the policy.
  §F4.2 filesystem policy is template-based (no machine paths); check_path
        denies traversal and symlink escape.
  §F4.3 no network client on the forward path.
  §F4.4 no shell / eval / exec / compile in aeon/ code paths.
  §F4.5 resource ceilings refuse over-limit configs.
  §F4.6 fail-closed conditions enumerated.

Torch-free.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_policy_exists_and_declares_execution_identity():
    from aeon.runtime_policy import load_policy
    pol = load_policy()
    ei = pol["execution_identity"]
    for key in ("requires_dedicated_non_admin", "no_package_install_authority",
                "no_unrestricted_shell", "no_credential_access",
                "may_alter_own_security_policy", "may_alter_own_executable_code",
                "explicit_output_directories"):
        assert key in ei, key
    # attestations that ADVANTAGE-INCREASE keys must be False
    assert ei["may_alter_own_security_policy"] is False
    assert ei["may_alter_own_executable_code"] is False


def test_policy_paths_are_templates_not_absolute():
    from aeon.runtime_policy import scan_no_absolute_paths_in_policy
    offenders = scan_no_absolute_paths_in_policy()
    assert not offenders, offenders


def test_check_path_denies_traversal():
    from aeon.runtime_policy import check_path, RuntimePolicyError
    subs = {"<repo>": os.getcwd(), "<corpus_root>": "/nonexistent",
            "<tokenizer_root>": "/nonexistent", "<tmp>": "/tmp",
            "<out_dir>": "runs/test"}
    # inside <repo>/docs — allowed for read
    p = check_path(os.path.join(os.getcwd(), "docs", "F1_THREAT_MODEL.md"), "read",
                   substitutions=subs)
    assert p.endswith("F1_THREAT_MODEL.md")
    # /etc/passwd — outside every root
    try:
        check_path("/etc/passwd", "read", substitutions=subs)
        assert False, "expected denial"
    except RuntimePolicyError as e:
        assert "outside allowed roots" in str(e)
    # relative traversal — refused
    try:
        check_path("docs/../../etc/passwd", "read", substitutions=subs)
        assert False, "expected denial"
    except RuntimePolicyError:
        pass


def test_check_path_denies_write_to_read_only():
    from aeon.runtime_policy import check_path, RuntimePolicyError
    subs = {"<repo>": os.getcwd(), "<corpus_root>": "/nonexistent",
            "<tokenizer_root>": "/nonexistent", "<tmp>": "/tmp",
            "<out_dir>": "runs/test"}
    try:
        check_path(os.path.join(os.getcwd(), "aeon", "hybrid.py"), "write",
                   substitutions=subs)
        assert False, "write allowed to source code"
    except RuntimePolicyError as e:
        assert "outside allowed roots" in str(e)


def test_check_path_denies_symlink_escape():
    from aeon.runtime_policy import check_path, RuntimePolicyError
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "repo", "docs"))
        # Place the "outside" target in a directory that is NOT in any
        # substitution — else the test's own <tmp>=/tmp overlaps and the
        # escape lands back inside an allow-listed root.
        outside_root = os.path.join(d, "outside_root")
        os.makedirs(outside_root)
        link = os.path.join(d, "repo", "docs", "escape")
        os.symlink(outside_root, link)
        subs = {"<repo>": os.path.join(d, "repo"),
                "<corpus_root>": "/does/not/exist/corpus",
                "<tokenizer_root>": "/does/not/exist/tokenizer",
                "<tmp>": "/does/not/exist/tmp",
                "<out_dir>": "runs/test"}
        try:
            check_path(link, "read", substitutions=subs)
            assert False, "symlink escape allowed"
        except RuntimePolicyError:
            pass


def test_no_network_client_in_forward_path():
    from aeon.runtime_policy import scan_forward_path_for_network_client
    offenders = scan_forward_path_for_network_client()
    # aeon/ and scripts/ (minus whitelisted subprocess for git rev-parse) must
    # not import network client modules.
    assert not offenders, offenders


def test_no_shell_or_eval_in_aeon():
    from aeon.runtime_policy import scan_for_shell_or_eval
    offenders = scan_for_shell_or_eval()
    assert not offenders, offenders


def test_resource_ceilings_refuse_over_limit_config():
    from aeon.runtime_policy import enforce_ceilings_on_config, RuntimePolicyError
    ok_model = {"K": 16}; ok_data = {"seq_len": 2048}; ok_train = {"batch_size": 4}
    enforce_ceilings_on_config(ok_model, ok_data, ok_train)      # OK
    for bad_data in ({"seq_len": 99999},):
        try:
            enforce_ceilings_on_config(ok_model, bad_data, ok_train)
            assert False, "seq_len ceiling not enforced"
        except RuntimePolicyError:
            pass
    for bad_train in ({"batch_size": 999},):
        try:
            enforce_ceilings_on_config(ok_model, ok_data, bad_train)
            assert False, "batch_size ceiling not enforced"
        except RuntimePolicyError:
            pass


def test_fail_closed_conditions_enumerated():
    from aeon.runtime_policy import fail_closed_conditions
    fcc = fail_closed_conditions()
    required = {"artifact_authentication_failure",
                "certificate_validation_failure",
                "protected_dtype_invariant_failure",
                "invalid_runtime_identity",
                "audit_write_failure",
                "policy_missing_or_incompatible",
                "unauthorized_network_capability_detected",
                "filesystem_escape_attempt",
                "critical_resource_ceiling_crossed",
                "checkpoint_integrity_failure"}
    missing = required - set(fcc)
    assert not missing, missing


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
