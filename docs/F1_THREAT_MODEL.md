# F1 — Threat Model (Aeon Defensive Resilience Upgrade)

**Sources of truth:** `docs/asset_registry.json`, `docs/threat_model.json`.
**Validator:** `aeon/policies/__init__.py`. **Enforcement:** `tests/test_threat_model.py` (7 checks).

Aeon protects the confidentiality (where applicable), **integrity**, **availability**, **authenticity**, and **provenance** of a defined set of assets under a defined set of adversaries. Every claim in this document is bounded to the evidence available under the F0–F9 upgrade — no claims about physical, firmware, or nation-state guarantees.

## Global non-guarantees

The following are OUT OF SCOPE for every claim in this document:

- Aeon does NOT claim resistance to a fully compromised processor.
- Aeon does NOT claim resistance to compromised firmware below the OS.
- Aeon does NOT claim resistance to live physical-memory extraction on general-purpose hardware.
- Aeon does NOT claim resistance to hostile hardware (malicious accelerator, backdoored NIC, etc.).
- Aeon does NOT claim guaranteed secret erasure from RAM.
- Aeon does NOT provide legal, safety, or military certification.

## Protected assets (27)

The full registry is in `docs/asset_registry.json`. Every asset lists confidentiality, integrity, availability, authenticity, provenance, recovery, rollback, retention, and authorized-disclosure requirements. The 27 assets cover source (A01), build (A02, A03, A04, A15), model (A05–A10, A18, A23), data (A11–A14), policy (A16, A17), telemetry (A19, A20, A24, A25), secrets (A21, A22), and operational (A26, A27) categories.

Requirement levels: `none` / `operator_asserted` / `structural` / `cryptographic` (or `cryptographic_optional` where confidentiality mode is opt-in).

## Adversaries (18)

| id | class | expected response | non-guarantee |
|---|---|---|---|
| T01 | accidental operator error | E1 config-invariant + strict_load reject; .prev retention | deliberate operator sabotage |
| T02 | malformed input | tokeniser byte-fallback + F4 input caps + bounded certificate | semantic content of output |
| T03 | poisoned corpus | F2 per-source manifest gate; F6 quarantine | subtle bias from a trusted source |
| T04 | unauthorised local user | F4 filesystem policy + no shell path | kernel privilege escalation |
| T05 | compromised dependency | F2 TCB report; pinned versions; weights_only load | supply chain of pinned deps |
| T06 | compromised update package | F2 provenance signature verification | stolen operator signing key |
| T07 | remote attacker | F4 zero-network local mode | operator-opened network surface |
| T08 | privileged host compromise | **FAIL CLOSED — out of scope** | TOTAL loss of runtime confidentiality/integrity |
| T09 | powered-off theft | F3 confidentiality mode (AEAD when key handle supplied) | keys stolen with device |
| T10 | checkpoint substitution | F3 sha256 + MAC + metadata gate | attacker who also controls sidecar without MAC |
| T11 | checkpoint rollback | F3.3 monotonic authorized_step | rollback with anti-rollback disabled |
| T12 | model extraction | F4 rate limits + F5 containment | patient sub-limit statistical extraction |
| T13 | resource exhaustion | F4 resource caps + F5 SAFE_HALT | authenticated raise of limits |
| T14 | audit tampering | F3 hash-chained audit | total rewrite by root |
| T15 | unauthorised config change | E1 tests + F3 strict_load bundle check | coordinated config + ckpt swap |
| T16 | recovery-state corruption | F3 recovery-artifact auth; F8 refuse-on-corrupt | all recovery paths destroyed |
| T17 | replay | F3.3 anti-rollback + F5 freshness | very recent within-window replay |
| T18 | incompatible-valid injection | F2 provenance chain + F3 bundle identity check | coordinately forged upstream bundle |

Each entry in `docs/threat_model.json` records: `access`, `knowledge`, `assets_at_risk`, `expected_response`, `residual_risk`, `non_guarantees`.

## Exit gate

- [x] Every protected asset is registered (`docs/asset_registry.json`, 27 entries covering the 25 directive categories).
- [x] Every required adversary is modelled (18 entries; `test_threat_model.py::test_threat_model_valid` verifies).
- [x] Every boundary has an explicit validation rule (see `F1_TRUST_BOUNDARIES.md`).
- [x] Residual risks and non-guarantees are stated.
- [x] No offensive capability introduced (documentation, schemas, validators only).
- [x] Inherited 61-check suite still passes (see F1 commit body).
- [x] New F1 tests pass (see F1 commit body).
