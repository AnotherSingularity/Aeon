# F9 — Defensive-Readiness Package

**Starting commit:** `fbdc1d9` (E-series close-out).
**F0–F8 additive ledger, working branch:** `claude/funny-cori-a3k5cf`.
**Final regression:** _recorded in the F9 commit body_.

## 1. Governing mission (§F9.1)

Aeon is an asymmetric defensive resilience platform. Its purpose is to preserve the functionality, integrity, availability, coordination, and recoverability of systems placed under its authorized defensive umbrella — with particular attention to constrained manufacturing and communications environments. Aeon is **not** an offensive platform. It may observe approved telemetry, analyze conditions, detect anomalies, recommend defensive actions, enter bounded degraded modes, and assist authenticated recovery. It must **not** become the sole authorization or execution authority for critical physical or communications operations.

## 2. Architecture preservation

Every architectural invariant from V0.02.03 (E-series) remains intact. See `docs/PRESERVATION_MANIFEST.md` (E0), `docs/TOPOLOGY_MAP.md` (E0), `docs/DEFINITION_OF_DONE.md` (E7). The F-series added defensive layers **around** these invariants without altering the certified computational topology.

The six V0.02.02 patches, K=16, fp32 Recursion, contractive certificate, substrate autonomy, and substrate-state dtype rule are enforced by named tests in the inherited 61-check suite and remain green after every F-phase.

## 3. Threat model + boundaries (§F1)

See `docs/F1_THREAT_MODEL.md`, `docs/F1_TRUST_BOUNDARIES.md`, `docs/asset_registry.json`, `docs/threat_model.json`, `docs/boundary_registry.json`. Non-guarantees explicitly stated (no processor / firmware / physical-memory / hostile-hardware claims).

## 4. Artifact provenance + TCB (§F2)

`docs/F2_ARTIFACT_PROVENANCE.md`, `docs/tcb_report.json`, `aeon/provenance.py`, `aeon/corpus_manifest.py`. Canonical identity is environment-portable; provenance chain covers Source → Build → Config → Tokenizer → Corpus → TrainingRun → Checkpoint → Evaluation → Recovery.

## 5. Protected checkpoint + key management (§F3)

`docs/F3_PROTECTED_CHECKPOINT.md`, `aeon/protected_checkpoint.py`, `aeon/audit.py`. HMAC-SHA256 envelope, optional AES-256-GCM confidentiality, monotonic `authorized_step` anti-rollback with explicit `RecoveryDecision`, hash-chained audit.

**Key-management assumptions and limitations:** production integrations replace `KeyRef.resolve` with a KMS or HSM callable. Aeon's development harness supplies an ephemeral in-memory key; that path is clearly labelled NOT PRODUCTION. Aeon does not claim guaranteed memory erasure on general-purpose hardware.

**Anti-rollback design:** monotone `authorized_step` recorded on save and enforced on load. Explicit `RecoveryDecision` (operator_authorization_ref, reason, current_state_identity, selected_state_identity, integrity_result, recovery_policy_version, resulting_authorized_state) is the only path to accept a lower state.

## 6. Runtime containment (§F4)

`docs/F4_RUNTIME_CONTAINMENT.md`, `aeon/runtime_policy.py`, `docs/runtime_policy.json`. Deny-by-default filesystem policy with template roots (no machine paths); zero required network in certified local mode; AST-verified absence of shell/eval/network-import in `aeon/`; resource ceilings; 10 named fail-closed conditions.

## 7. Defensive continuity (§F5)

`docs/F5_CONTINUITY_FRAMEWORK.md`, `aeon/continuity.py`. Seven-state deterministic machine; manufacturing + communications ANALYTICAL abstractions (recommendations only); graceful-degradation order that preserves security/integrity first.

**Manufacturing analytical boundary:** `analyze_manufacturing_telemetry` observes and recommends. No vendor commands, no direct control protocols.

**Communications analytical boundary:** `analyze_comms_telemetry` observes and recommends. No interception, jamming, exploitation, credential capture, covert access, or protocol manipulation.

## 8. Adversarial resilience (§F6)

`docs/F6_ADVERSARIAL_RESILIENCE.md`, `docs/f6_adversarial_results.json`, `tests/test_adversarial.py`. Twenty cases across artifact / data / runtime / model-state / availability categories; every case records threat_id, precondition, injection, expected/actual, detection/containment/recovery, audit_event_id, pass/fail.

## 9. Protected efficiency (§F7)

`docs/F7_PROTECTED_EFFICIENCY.md`, `docs/f7_evidence.json`, `scripts/f7_certify.py`. Six profiles measured with mandatory protection active; costs separated (base / observability / artifact_verification / cryptographic / audit / runtime_containment / recovery / total). No single unexplained percentage.

## 10. Recovery report (§F8)

`docs/F8_RECOVERY_REPORT.md`, `docs/f8_evidence.json`, `scripts/f8_recovery.py`. Thirteen exercises; every §F8.2 field recorded per exercise; §F8.4 failure policy enforced. Zero policy violations at close.

## 11. Deployment assumptions

- Aeon's OS-level containment (seccomp / AppArmor / cgroups / systemd `NoNewPrivileges`) is deployment work, not Aeon-side code.
- Production key management is via KMS/HSM, injected through `KeyRef.resolve`.
- Corpus + tokenizer paths in `configs/aeon_350m_primary.yaml` are placeholders; the operator populates them at launch. Their identities become part of the checkpoint bundle and cannot silently drift.
- Aeon does not have network authority in certified local mode. Any future connected mode requires an independent authorization pass — not included in this upgrade.

## 12. Public description

> Aeon is a defensively-designed architecture with two independent computational streams — a recurrent substrate and a transformer — integrated through a contractive slow-clock Recursion mechanism whose σ<margin certificate is structural. It ships with authenticated, atomic, rollback-resistant checkpoints; a deny-by-default runtime posture; hash-chained audit records; and a seven-state continuity machine for observing manufacturing and communications environments. Aeon does not act as a control system and it does not provide offensive capability.

## 13. Controlled engineering description

Aeon V0.02.03 (post-F9) is a 350.28 M-parameter model composed of:

- `AeonTransformer` (24 layers × 1024 hidden × 16 heads with GQA=4, SwiGLU, rotary; fp32 rope frequencies computed fresh per forward).
- `MatrixStateCell` substrate (H=8 heads × N=64) with an adaptive-feedback controller (bound-preserving stressed-mode blend; fp32 gate scalars).
- `RecursionJoiner` — Cayley + σ<MARGIN certificate, fp32, one broadcast per K-window, K=16.

Every checkpoint carries schema_version, patch_manifest_version, K, source_commit, tokenizer identity, corpus identity, precision policy, certificate policy, runtime policy, security policy, HMAC tag, `authorized_step`, and (optionally) AEAD ciphertext. Every training / inference / diagnostic / recovery command is bounded by the F4 runtime policy.

## 14. Operations & recovery guide

See §17 (verified commands) plus the E-series `docs/OPERATIONS.md` for the primary training campaign procedures. F8 recovery patterns supplement §3 of that guide.

## 15. Efficiency claim boundaries

Approved framing after F7:

> Aeon demonstrates measured protected efficiency at its current implementation scale on laptop-class CPU hardware.

The following claims are OUT OF BOUNDS: frontier superiority, universal nation-state resistance, military certification, FLOP-efficiency claims not backed by measured hardware, energy-efficiency claims without energy measurements, full-scale performance without matched large-scale comparison.

## 16. Known limitations

- No production key manager is bundled. Confidentiality-mode encryption uses `cryptography.hazmat`'s AES-GCM; production integrations must supply a real key backend and inject via `KeyRef.resolve`.
- OS-level containment (seccomp / AppArmor / cgroups / systemd hardening) is deployment work.
- No CI/CD workflow is configured in this repository; every "workflow" reported below is either N/A (nothing was configured) or has terminal status "completed" (local tests).
- Efficiency measurements at F7 were taken on 4-thread CPU with a small representative model. Production-scale numbers will differ.
- The F7 cost deltas at the small scale are noise-dominated per step; the report says so explicitly.
- Aeon is not connected to real manufacturing or communications equipment by this upgrade; the F5 abstractions are analytical only.

## 17. Verified commands (§F9.3)

All paths repo-relative; `<repo>` is the working tree.

```bash
# Full integrity + regression verification
export PYTHONPATH=<repo>
for t in test_substrate_port test_aeon_sanity test_tokenizer test_feedback \
         test_feedback_diagnostics test_six_patches test_recursion_topology \
         test_stream_independence test_config_invariants test_observability \
         test_checkpoint test_diagnose test_threat_model test_provenance \
         test_protected_checkpoint test_runtime_policy test_continuity \
         test_adversarial; do
  python3 tests/$t.py
done

# Fresh local training (E7 primary path preserved)
python scripts/train.py --config configs/aeon_350m_primary.yaml

# Protected checkpoint save (in-code)
# → happens automatically when a protected_save path is wired into scripts/train.py
#   for confidentiality-mode deployments (F3 API is aeon.protected_checkpoint).

# Protected resume
python scripts/train.py --config configs/aeon_350m_primary.yaml    # train.resume=true

# Validation only  (single forward + loss + audit; no optimizer step)
# → scripts/diagnose.py --subcommand certificate + gradients gives this without training

# Offline diagnostics
python scripts/diagnose.py --config configs/aeon_350m_primary.yaml \
    --ckpt runs/aeon_350m_primary/ckpt_1000.pt --subcommand all

# Generation test
python scripts/infer.py --config configs/aeon_350m_primary.yaml \
    --ckpt runs/aeon_350m_primary/ckpt_1000.pt --tokenizer tokenizer/aeon.model \
    --prompt "Aeon" --max-new-tokens 64

# Adversarial suite
python tests/test_adversarial.py

# Recovery exercise
python scripts/f8_recovery.py

# Runtime-policy verification
python tests/test_runtime_policy.py

# Provenance verification
python tests/test_provenance.py

# Full defensive-efficiency certification
python scripts/f7_certify.py
```

## 18. Machine-readable evidence bundle

`docs/f9_final_evidence.json` — collects identities, ledger, results, cost breakdowns, recovery measurements, and open limitations for the F9 close-out.

## 19. Final regression

Recorded in `docs/f9_final_evidence.json::final_regression`.

## 20. Definition of done (§F9.5)

`docs/F9_DEFINITION_OF_DONE.md`.
