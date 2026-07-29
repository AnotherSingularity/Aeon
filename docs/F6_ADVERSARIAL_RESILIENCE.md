# F6 — Authorized Adversarial Resilience

**Sources of truth:** `aeon/adversarial.py`, `tests/test_adversarial.py`, `docs/f6_adversarial_results.json`.

Aeon's defensive test harness attempts to violate integrity, provenance, containment, availability, and recovery rules by driving the SAME public defensive interfaces an operator would (`strict_load`, `protected_load`, `refuse_if_invalid`, `check_path`, `enforce_ceilings_on_config`, `analyze_*`) — and observing that they refuse. **The harness is defensive: it does not build offensive capability**; it exercises existing rejection paths.

## Case format (§F6.6)

Each `AdversarialCase` records the required fields:

- `threat_id` (T01–T18 from `docs/threat_model.json`)
- `category` (`artifact` / `data` / `runtime` / `model_state` / `availability`)
- `precondition`
- `injection` (the hostile condition)
- `expected_response`
- `actual_response`
- `detection` / `containment` / `recovery`
- `audit_event_id`
- `passed`

Machine-readable results: `docs/f6_adversarial_results.json`, regenerated every test run.

## Cases

### §F6.1 Artifact attacks (7)

| case | threat | expected response |
|---|---|---|
| `modified_checkpoint_bytes` | T10 | sha256 gate → `CheckpointCorrupt` |
| `replaced_weight_tensor` | T10 | sha256 / MAC refuses swap |
| `changed_tokenizer_vocab_mismatch` | T18 | `strict_load` → `CheckpointIncompatible: vocab_size mismatch` |
| `changed_patch_manifest` | T15 | `strict_load` → `patch_manifest_version mismatch` |
| `missing_authentication_metadata` | T10 | `protected_load` → `CheckpointCorrupt: missing envelope metadata` |
| `corrupted_audit_chain` | T14 | `verify_chain` names the first inconsistency |
| `unauthorized_older_checkpoint` | T11 | `protected_load` → `AntiRollbackViolation` |

### §F6.2 Data attacks (3)

| case | threat | expected response |
|---|---|---|
| `malformed_corpus_manifest` | T02 | `refuse_if_invalid` → `ProvenanceError` |
| `quarantined_source_smuggled_into_train` | T03 | `ProvenanceError: quarantined ...` |
| `content_sha256_mismatch` | T02 | `verify_source_content` → `ProvenanceError` |

### §F6.3 Runtime attacks (5)

| case | threat | expected response |
|---|---|---|
| `path_traversal` | T15 | `check_path('/etc/passwd', 'read')` → `RuntimePolicyError` |
| `symlink_escape` | T15 | symlink whose realpath is outside allow-list refused |
| `no_shell_or_eval_call_in_aeon` | T04 | static AST scan finds no offender |
| `no_network_client_import` | T07 | static AST scan finds no offender |
| `over_limit_seq_len` | T13 | `enforce_ceilings_on_config` → `RuntimePolicyError` |

### §F6.4 Model-state attacks (3)

| case | threat | expected response |
|---|---|---|
| `certificate_forced_violation` | T15 | `audit()['holds']` becomes `False` when `_build` is bypassed |
| `recursion_stays_fp32_after_cast` | T15 | every recursion param remains `fp32` post `model.to(bf16)` |
| `K_config_drift_detected` | T15 | E1 config-invariant assertion caught inline |

### §F6.5 Availability attacks (2)

| case | threat | expected response |
|---|---|---|
| `interrupted_write_preserves_prev` | T16 | prior checkpoint sha256 unchanged after `torch.save` raises mid-write |
| `corrupted_latest_valid_prev` | T16 | `strict_load` refuses corrupted latest; `.prev` still on disk |

## Results

Latest run: **20 / 20 pass, 0 fail**. Every case exercised the defensive interface, produced the expected refusal, and recorded the audit event id. See `docs/f6_adversarial_results.json` for the machine-readable evidence bundle.

## Exit gate

- [x] All defined adversarial cases produce the expected defensive response.
- [x] Failures are reproducible (fixed seeds where applicable, or structural refusals).
- [x] Audit evidence is complete (each case has a UUID audit_event_id + JSON record).
- [x] The harness remains local and bounded (no network, no shell, no exec).
- [x] No offensive, intrusive, or unauthorised capability is introduced (only refusal-observation).
- [x] Suite: 117 inherited + 20 F6 = **137/137 pass.**
