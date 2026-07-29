# F1 — Trust Boundaries (Aeon Defensive Resilience Upgrade)

**Source of truth:** `docs/boundary_registry.json`. **Validator:** `aeon/policies/__init__.py`.

Aeon defines 16 named boundaries. Every crossing declares: permitted input, required identity, validation procedure, size/resource limits, failure behaviour, audit requirement, whether it can influence model state, and whether it can influence operational authority.

## Boundaries

| id | boundary | may influence model state | may influence operational authority | validation summary |
|---|---|:---:|:---:|---|
| B01 | corpus_ingestion | yes | no | UTF-8 records ≤ 1 MiB; source-manifest gate; unidentified sources rejected |
| B02 | tokenizer | yes | no | SentencePiece + fixed-id layout; F3 strict_load cross-check |
| B03 | training_process | yes | no | E1 config-invariant tests + F2 provenance verification |
| B04 | model_runtime | no | no | dtype+shape checks; bounded output_bound; no input-code path |
| B05 | recursion_state | yes | no | fp32; certificate audit; structural cadence |
| B06 | checkpoint | yes | no | sha256 + MAC + patch/manifest/source_commit bundle |
| B07 | key_management | no | yes | handle-only; keys never touch disk in cleartext |
| B08 | operator | no | yes | F5 signed operator-decision + deterministic authorization outside model |
| B09 | evaluation | no | no | partition-leakage + evaluation-contamination checks |
| B10 | network | no | no | ZERO required network in certified local mode; F4 verification |
| B11 | filesystem | yes | no | F4 path allowlist + traversal/symlink denial |
| B12 | update | yes | yes | operator signature verify + strict_load bundle drift refused |
| B13 | recovery | yes | yes | F3.3 authorized rollback record with every field present |
| B14 | manufacturing_interface | no | no | F5 analytical; freshness+sequence+authenticity; NEVER control |
| B15 | communications_interface | no | no | F5 analytical; freshness+sequence+replay indicators; NEVER control |
| B16 | audit | no | no | hash-chained events; SAFE_HALT on write failure |

## Failure behaviour glossary

- **reject_and_audit** — refuse the operation, write an audit event.
- **refuse_load** — refuse to bring the artefact into runtime; do not partially load.
- **abort_before_start** — training process is not launched; no state altered.
- **fail_closed_on_cert_violation** — stop; do not weaken margins to continue.
- **refuse_and_preserve_prev** — reject the new artefact and keep the `.prev` known-good.
- **quarantine_frame** — F5 telemetry frame is moved to a quarantine queue for operator review.
- **safe_halt_on_audit_write_failure** — F5 SAFE_HALT if the audit backend cannot record events.
- **fail_closed_on_any_attempt** — F4 network attempt refused and audited.

## Influence flags

Only boundaries whose `may_influence_operational_authority` is true carry operational-authority weight (B07 key management, B08 operator, B12 update, B13 recovery). Aeon is never the SOLE authorization or execution authority — every such boundary requires an external operator identity per the mission clarification.

## Exit gate

- [x] All 16 required boundaries encoded (`docs/boundary_registry.json`).
- [x] Each declares permitted input, required identity, validation, size limits, failure behaviour, audit, and both influence flags.
- [x] Validated by `tests/test_threat_model.py::test_boundary_registry_valid` + `test_every_boundary_has_influence_flags`.
