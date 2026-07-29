# F9 — Definition of Done (Defensive Resilience Upgrade)

Directive §F9.5 enumerates 34 conditions. Each row identifies the evidence.

| # | condition | evidence | status |
|---|---|---|---|
| 1 | All inherited Aeon invariants remain intact | `docs/DEFINITION_OF_DONE.md` (E-series) + full E-suite still passes | **DONE** |
| 2 | The F1 threat model is complete | `docs/threat_model.json` (18 adversaries) + `tests/test_threat_model.py` | **DONE** |
| 3 | Trust boundaries are enforced | `docs/boundary_registry.json` (16 boundaries) + F1 validators | **DONE** |
| 4 | Artifact identities are portable and reproducible | `aeon/provenance.py::canonicalize` + `tests/test_provenance.py` (env-portable, order-agnostic) | **DONE** |
| 5 | Provenance links source through recovery | `aeon/provenance.py::CHAIN_KINDS` covers Source→Build→Config→Tokenizer→Corpus→TrainingRun→Checkpoint→Evaluation→Recovery | **DONE** |
| 6 | Unauthenticated checkpoints are rejected | `aeon/protected_checkpoint.py::protected_load` MAC gate + `tests/test_protected_checkpoint.py` | **DONE** |
| 7 | Corrupted checkpoints are rejected | sha256 sidecar + MAC; F6 case `modified_checkpoint_bytes`; F8 exercise 1 | **DONE** |
| 8 | Unauthorized rollback is rejected | `AntiRollbackViolation`; F6 case `unauthorized_older_checkpoint`; F8 exercise 3 | **DONE** |
| 9 | Authorized recovery is supported | `RecoveryDecision`; F8 exercise 4 (authorized rollback) + 11 (restore from `.prev`) | **DONE** |
| 10 | Runtime authority is deny-by-default | `docs/runtime_policy.json` + `aeon/runtime_policy.py::check_path` + `tests/test_runtime_policy.py` | **DONE** |
| 11 | Certified local mode requires no network | AST scan `scan_forward_path_for_network_client`; F6 case `no_network_client_import` | **DONE** |
| 12 | Filesystem access is bounded | `check_path` refuses traversal + symlink escape + writes to read-only; F6 cases | **DONE** |
| 13 | Arbitrary code and process execution are unavailable | AST scan `scan_for_shell_or_eval` (only allowlisted dynamic import in `provenance.py`) | **DONE** |
| 14 | Resource controls are enforced | `enforce_ceilings_on_config`; F6 case `over_limit_seq_len` | **DONE** |
| 15 | Aeon cannot increase its own authority | F5 `ContinuityController` refuses `aeon_analytical` initiator for consequential positive transitions; policy attest `may_alter_own_security_policy=false` | **DONE** |
| 16 | Continuity-state transitions are deterministic | `TRANSITION_TABLE` + tests | **DONE** |
| 17 | Manufacturing and communications support remains analytical | `analyze_*` returns observations only; no vendor commands/protocols in source | **DONE** |
| 18 | Artifact adversarial tests pass | F6 §F6.1 — 7 cases all pass | **DONE** |
| 19 | Data adversarial tests pass | F6 §F6.2 — 3 cases all pass | **DONE** |
| 20 | Runtime-containment tests pass | F6 §F6.3 — 5 cases all pass | **DONE** |
| 21 | Model-state integrity tests pass | F6 §F6.4 — 3 cases all pass | **DONE** |
| 22 | Availability tests pass | F6 §F6.5 — 2 cases all pass | **DONE** |
| 23 | Recovery exercises pass | F8 — 13/13 exercises pass | **DONE** |
| 24 | Protected performance is measured | `scripts/f7_certify.py` + `docs/f7_evidence.json` | **DONE** |
| 25 | Security costs are separated | F7 cost categories separated (base / obs / verify / crypto / audit / containment / recovery / total) | **DONE** |
| 26 | Mandatory protections remain enabled during measurement | `f7_evidence.json::protection_envelope_active_during_measurement` all true | **DONE** |
| 27 | Public and controlled descriptions are truthful | `docs/F9_DEFENSIVE_READINESS.md` §12–13 | **DONE** |
| 28 | No offensive capability is introduced | Explicit exclusion list in F1/F5/F6 docs; source contains no vendor commands, no interception, no jamming | **DONE** |
| 29 | No host-specific evidence paths remain | F0 hygiene fix + F7/F8 evidence scrub (`/home/user/AeonV0.02` → `<repo>`) | **DONE** |
| 30 | Documentation matches code | Every doc references code by file path; every code addition has a docs entry | **DONE** |
| 31 | All workflows reach terminal status | Repository has no `.github/` and no configured CI; terminal status N/A recorded in F0 | **DONE** |
| 32 | The complete suite passes | recorded in `docs/f9_final_evidence.json::final_regression` | **DONE** |
| 33 | No unresolved architecture / integrity / containment / recovery / runtime blocker | F0–F8 phases all closed PASS; F9 final regression records no fail | **DONE** |
| 34 | The final commit and closure report are pushed | recorded in the F9 commit body + push output | **DONE** |

**Verdict: DONE — 34 / 34.**
