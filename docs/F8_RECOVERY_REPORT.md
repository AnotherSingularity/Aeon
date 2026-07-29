# F8 — Recovery Exercise Report

**Runner:** `scripts/f8_recovery.py`. **Evidence:** `docs/f8_evidence.json`.

Aeon detects failure, contains it, refuses corrupted state, and returns to an authenticated known-good state without losing its certified architecture. Every §F8.1 scenario is executed; every §F8.2 field is recorded per exercise; every §F8.4 failure policy is enforced.

## 1. Exercises (§F8.1) — 13 / 13 pass

| # | exercise | detection | containment | recovery | K after | recursion fp32 | cert |
|---|---|---|---|---|:-:|:-:|:-:|
| 1 | corrupted newest checkpoint | detected | contained | possible (via `.prev`) | 16 | ✓ | ✓ |
| 2 | interrupted checkpoint save | detected | contained | not required (prior intact) | — | — | — |
| 3 | unauthorized rollback | detected | contained | requires operator decision | — | — | — |
| 4 | explicit authorized rollback | n/a | n/a | possible | 16 | ✓ | ✓ |
| 5 | invalid runtime configuration | detected | contained | fix config | — | — | — |
| 6 | missing required artefact (envelope `.meta.json`) | detected | contained | restore meta | — | — | — |
| 7 | contractive certificate failure | detected | n/a | not required (fail-closed) | — | — | ✗ (forced) |
| 8 | resource-exhaustion → SAFE_HALT | detected | contained | operator restart | — | — | — |
| 9 | audit-output failure | detected | contained | operator | — | — | — |
| 10 | abrupt termination during training | n/a | contained | possible | 16 | ✓ | ✓ |
| 11 | restore from `.prev` | n/a | n/a | possible | 16 | ✓ | ✓ |
| 12 | provenance mismatch refused | detected | contained | restore provenance | — | — | — |
| 13 | protected-state auth failure (wrong MAC key) | detected | contained | use correct key | — | — | — |

Every exercise records: `detection_time_s`, `recovery_time_s` (where applicable), `audit_chain_ok`, and the architecture-manifest / six-patch-manifest / K / dtype / certificate post-recovery state per §F8.2. See `docs/f8_evidence.json` for the full record.

## 2. Recovery measurements (§F8.3)

Detection times sit in the **micro-to-millisecond** band for corruption / rollback / metadata refusal. Recovery times for the two "restore" exercises (corrupted-latest via `.prev`, restore-from-`.prev`) are ~10 ms — dominated by the `strict_load` payload read of the tiny model. See `docs/f8_evidence.json` for the exact per-exercise timings; production numbers will scale with checkpoint size.

## 3. Failure policy (§F8.4) — enforced

The runner enforces `passed=False` on any of:

- A corrupted artefact becomes active.
- An unauthorised old checkpoint becomes active.
- Architecture / K / dtype changes silently.
- The certificate is skipped.
- Audit continuity is silently lost (`audit_chain_ok=False`).
- Recovery requires disabling mandatory controls.

Latest run: **0 policy violations.** Audit chains verified across every exercise directory.

## 4. Exit gate

- [x] Every required scenario executed.
- [x] Known-good restoration succeeds where recovery is possible (exercises 1, 4, 10, 11).
- [x] Unsafe restoration is rejected (exercises 3, 5, 6, 12, 13).
- [x] Architecture and certificate invariants survive recovery (K=16, recursion fp32, certificate hold — recorded per exercise).
- [x] Audit continuity preserved (`audit_chain_ok=True` on every exercise) or explicitly fails closed (exercise 9).
- [x] All recovery tests pass: **13 / 13.**
