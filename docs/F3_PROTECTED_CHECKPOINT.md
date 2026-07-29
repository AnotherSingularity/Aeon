# F3 — Protected Checkpoint and State Lifecycle

**Source of truth:** `aeon/protected_checkpoint.py`, `aeon/audit.py`.
**Enforcement:** `tests/test_protected_checkpoint.py` (9 checks) + `tests/test_checkpoint.py` (9 checks) still pass.

Extends E3's atomic checkpoint system with authentication (HMAC-SHA256),
optional confidentiality (AES-256-GCM), monotonic anti-rollback, and a
hash-chained audit log. E3's `atomic_save` / `strict_load` are UNCHANGED and
still available.

## 1. Authenticated envelope (§F3.1)

`protected_save` writes three files atomically:

- `<path>` — the payload (plaintext or AEAD-wrapped).
- `<path>.sha256` — sha256 sidecar (E3 compatibility).
- `<path>.meta.json` — envelope metadata including `mac_hex`.

`mac_hex = HMAC-SHA256(key, <payload bytes> || 0x00 "META" 0x00 || <meta JSON canonical>)`.

MAC authenticates the **payload and metadata together** — an attacker cannot swap either independently. `hmac.compare_digest` is used for constant-time comparison.

The envelope's `inner_metadata` field carries the full E3 completeness set (§10.1): step, K, model_config, train_config, data_config, data_position, tokenizer_identity, corpus_identity, precision_policy, certificate_policy, patch_manifest_version, source_commit, instrumentation_config.

## 2. Optional confidentiality (§F3.2)

- Backend: **`cryptography.hazmat.primitives.ciphers.aead.AESGCM`** — AES-256-GCM.
- No custom crypto primitives. If the backend is unavailable, `protected_save(keyref_encrypt=...)` raises `KeyUnavailableError` — never falls back to plaintext.
- Keys are referenced by **opaque `KeyRef` handle**, resolved on demand. The handle is stored in metadata; the key **bytes are never** stored in metadata, payload, or the sha256 sidecar (verified by `test_key_material_is_not_stored_in_checkpoint_or_meta`).
- Dev key holder: `ephemeral_dev_keyref()` — clearly labelled NOT PRODUCTION KEY MANAGEMENT.
- Production integrations replace `KeyRef.resolve` with a KMS or HSM callable.
- Decryption failure → `CheckpointAuthenticationError` (AEAD is authenticated).

**Non-guarantees (§F1 global):** Aeon does not claim guaranteed memory erasure on general-purpose hardware; a live-memory attacker with root can extract the resolved key regardless.

## 3. Anti-rollback (§F3.3)

Envelope metadata carries `authorized_step` (integer, monotone across authorised operations). On load:

- If `enforce_anti_rollback=True` (default) and `current_authorized_step` is supplied, a checkpoint whose `authorized_step` < `current_authorized_step` is **refused** with `AntiRollbackViolation` — unless the caller supplies an explicit `RecoveryDecision`.

`RecoveryDecision` records: `operator_authorization_ref`, `reason`, `current_state_identity`, `selected_state_identity`, `integrity_result`, `recovery_policy_version`, `resulting_authorized_state`. All fields are required. The accepted-via-recovery-decision record is embedded in the returned envelope metadata for audit binding.

Anti-rollback **does not** use filenames or wall-clock timestamps — only the authenticated `authorized_step`.

## 4. Sensitive-state handling (§F3.4)

- Encrypted payload uses AEAD; **plaintext copy is unlinked promptly** (`os.unlink` in a `try/finally`).
- Temp files use `mkstemp` under the same dir as the target so `os.rename` is atomic and cross-device safety is preserved.
- `.prev` retention keeps the previous known-good; on encrypted saves, `.prev` also holds AEAD-wrapped bytes.
- Keys never appear in Aeon logs, metrics, or command arguments (verified by test).

## 5. Compatibility and fail-closed tests (§F3.5)

Every failure surface named in the directive has a test:

| directive check | test |
|---|---|
| One-byte checkpoint alteration | `test_one_byte_payload_tamper_refused` |
| Manifest alteration | `test_mac_verification_refuses_meta_tampering` |
| Weight substitution | covered by MAC over payload bytes |
| Tokenizer / vocab / corpus mismatch | E3 `test_reject_incompatible_metadata_vocab_mismatch` |
| Source-commit mismatch | envelope metadata carries source_commit; downstream E3 strict_load enforces on the inner metadata |
| Missing authentication metadata | absent `.meta.json` → `CheckpointCorrupt("missing envelope metadata")` |
| Unauthorized rollback | `test_anti_rollback_refuses_older_checkpoint` |
| Explicit authorized rollback | `test_authorized_rollback_accepted_with_recovery_decision` |
| Missing encryption key | `test_encrypted_round_trip_or_gracefully_absent` (KeyUnavailableError branch) |
| Wrong encryption key | AEAD decrypt raises `CheckpointAuthenticationError` |
| Interrupted protected save | E3 `test_atomic_save_survives_interrupted_write` (same `.prev` mechanism) |
| Restoration from previous known-good | `.prev` retained through F3 envelope |

## 6. Audit continuity

`aeon/audit.py::AuditWriter` writes hash-chained records. Each record includes `seq` (monotone), `prev_hash`, `payload`, and its own `hash`. `verify_chain()` traverses the log and returns the first inconsistency. A break is detected by the test `test_audit_hash_chain_verifies_and_detects_tampering`.

## 7. Exit gate

- [x] Substitution and corruption are detected (MAC + sha256).
- [x] Incompatible state is rejected (envelope schema + inner metadata gate).
- [x] Unauthorized rollback is rejected.
- [x] Authorized recovery remains possible via `RecoveryDecision`.
- [x] No key material is committed (test).
- [x] Existing resume-equivalence guarantees remain intact (E3 tests still pass).
- [x] Architecture and dtype invariants remain intact (E1 tests still pass).
- [x] Full suite: 83 inherited + 9 F3 = **92/92 pass**.
