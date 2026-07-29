# F4 — Runtime Containment

**Sources of truth:** `aeon/runtime_policy.py`, `docs/runtime_policy.json`.
**Enforcement:** `tests/test_runtime_policy.py` (9 checks).

Aeon is a **deny-by-default** citizen of its host. Compromise of model output must not automatically become compromise of the host or the protected environment. This document names the Aeon-side attestations; **OS-level containment (seccomp, AppArmor, unit-file `NoNewPrivileges`, restricted capabilities, cgroups) is the deployment's responsibility** and is not something Aeon can enforce for the operator.

## 1. Execution identity (§F4.1)

Recorded in `docs/runtime_policy.json::execution_identity`:

- `requires_dedicated_non_admin: true`
- `no_package_install_authority: true`
- `no_unrestricted_shell: true`
- `no_credential_access: true`
- `may_alter_own_security_policy: false`
- `may_alter_own_executable_code: false`
- `explicit_output_directories: true`

Aeon-side attestation: the runtime code does not read credentials, does not modify the policy file, and does not rewrite its own executable code (verified indirectly by the scanners and by the atomic-save mechanism operating only on approved output paths).

## 2. Filesystem policy (§F4.2)

Template roots (no machine-specific paths — verified by `test_policy_paths_are_templates_not_absolute`):

**Read-only:** `<repo>/aeon`, `<repo>/scripts`, `<repo>/configs`, `<repo>/docs`, `<corpus_root>`, `<tokenizer_root>`.

**Writable:** `<repo>/runs/<out_dir>`, `<repo>/runs/<out_dir>/audit`, `<tmp>`.

`aeon.runtime_policy.check_path(path, mode, substitutions=…)`:

- Rejects path traversal (`..` in the path).
- Rejects symlink escapes (compares `realpath` of the child against `realpath` of every root — verified by `test_check_path_denies_symlink_escape`).
- Rejects writes to read-only roots (`test_check_path_denies_write_to_read_only`).

## 3. Network policy (§F4.3)

**Certified local mode has ZERO required network use.**

`scan_forward_path_for_network_client` — AST scan of `aeon/` and `scripts/` — must find no import of `socket`, `urllib.request`, `requests`, `http.client`, `asyncio.open_connection`, `smtplib`. String literals and comments are ignored (test at `test_no_network_client_in_forward_path`).

Future connected mode is out of scope for this upgrade. Adding one would require defining an endpoint allowlist, mutual authentication, replay protection, message-size bounds, rate limits, timeouts, and audit requirements — none of which is present in current code.

## 4. Process and code-execution policy (§F4.4)

`scan_for_shell_or_eval` — AST scan of `aeon/` — must find no call to:

- `os.system(...)`, `os.popen(...)`
- `subprocess.Popen(...)`, `subprocess.check_call(...)`, `subprocess.call(...)`
- `eval(...)`, `exec(...)`, `compile(...)`
- `__import__(...)` (unless allowlisted with a documented reason)

Documented allowlist (`_ALLOWED_DYNAMIC_IMPORT`):

- `aeon/provenance.py::__import__` — runtime version reporting for a fixed known dep set (torch, sentencepiece, yaml, numpy). The names are hardcoded string literals, not derived from any input.

`scripts/` is permitted to use `subprocess.run` for `git rev-parse HEAD` (source-commit identity) and for spawning test/diagnostic entry points; these are not model-directed and are audited by inspection.

## 5. Resource controls (§F4.5)

From `docs/runtime_policy.json::resource_controls`, enforced by `enforce_ceilings_on_config`:

| control | ceiling |
|---|---:|
| `seq_len_max` | 8192 |
| `batch_size_max` | 32 |
| `input_size_bytes_max` | 1 048 576 (1 MiB per record) |
| `disk_ceiling_gb_per_run` | 64 |
| `checkpoint_retention_max` | 20 |
| `audit_log_ceiling_mb` | 4096 |
| `metrics_log_ceiling_mb` | 4096 |
| `evaluation_duration_s_max` | 3600 |
| `diagnostic_duration_s_max` | 900 |
| `queue_depth_max` | 128 |
| `restart_rate_per_hour_max` | 8 |
| `temporary_file_count_max` | 32 |

CPU utilisation is controlled through the OS-standard `OMP_NUM_THREADS`. Memory ceiling is deployment-configured (cgroups).

## 6. Fail-closed conditions (§F4.6)

Named list in `docs/runtime_policy.json::fail_closed_conditions`, exposed via `fail_closed_conditions()`:

- artifact_authentication_failure
- certificate_validation_failure
- protected_dtype_invariant_failure
- invalid_runtime_identity
- audit_write_failure
- policy_missing_or_incompatible
- unauthorized_network_capability_detected
- filesystem_escape_attempt
- critical_resource_ceiling_crossed
- checkpoint_integrity_failure

Each maps to an existing test in F1–F4 or an F5 SAFE_HALT trigger.

## 7. Exit gate

- [x] Deny-by-default authority mechanically enforced by `check_path`; OS-level enforcement documented as deployment responsibility.
- [x] Certified local mode operates without network access (AST scan).
- [x] Filesystem escape attempts fail (test).
- [x] Arbitrary process and code execution are unavailable (AST scan).
- [x] Resource controls behave deterministically (test).
- [x] Fail-closed conditions enumerated and named.
- [x] Suite: 92 inherited + 9 F4 = **101/101 pass.**
