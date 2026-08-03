# DESKTOP-R0 — Evidence Audit of the Prior Closure Claim

**Audited head:** `d91a836`
**Ladder verdict downgrade:** `FUNCTIONAL_UNPACKAGED_DESKTOP` → **`FUNCTIONAL_LOCAL_BUILD`**

The prior closure honestly declared State B (functional local build,
Windows blocker unmet). But some of its individual "PASS" labels
combined static and dynamic evidence in ways that made the aggregate
look stronger than the underlying raw evidence. This audit
reclassifies each claim.

Full machine-readable version: `docs/desktop/desktop_r0_evidence_audit.json`.

---

## 1. Reports listed in §41 that were never actually written

* `docs/desktop/DESKTOP_ACCEPTANCE_REPORT.md` — MISSING at `d91a836`.
* `docs/desktop/DESKTOP_SOAK_REPORT.md` — MISSING at `d91a836`.

These will be written during the R0…R5 tranches with actual evidence
attached.

---

## 2. Per-claim reclassification

| Prior claim                                | Prior label                          | Audit verdict                    | Reason for downgrade                                                                                                   |
| ------------------------------------------ | ------------------------------------ | -------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Logit equivalence**                      | BYTE_IDENTICAL (`torch.equal`)       | **PARTIALLY_PROVEN**             | Only PARAMETER equality was proven; no forward-logit comparison was run.                                               |
| **Deterministic generation equivalence**   | asserted via param equality          | **PARTIALLY_PROVEN**             | Same-parameters + same-config would produce same logits, but no explicit token-sequence equality test was executed.    |
| **Transformer execution (per generation)** | asserted                             | **PARTIALLY_PROVEN**             | Presence of transformer parameters proven; per-token execution trace not captured.                                     |
| **Substrate execution (per generation)**   | asserted                             | **PARTIALLY_PROVEN**             | Same as above.                                                                                                         |
| **Fixed-K boundary schedule**              | HybridModel.K == 16                  | **PARTIALLY_PROVEN**             | K attribute verified; per-token boundary-count trace not captured.                                                     |
| **Shared-broadcast cardinality**           | HybridModel.py unchanged             | **PARTIALLY_PROVEN**             | Source not modified; no per-generation broadcast-count trace.                                                          |
| **ACIS OFF**                               | shuttle=None in _generate            | **PROVEN**                       | Runtime literally passes shuttle=None into HybridModel.forward.                                                        |
| **Streaming**                              | events fire during generation        | **PROVEN**                       | test_runtime_generates_bounded_tokens_end_to_end verifies TOKEN_GENERATED + TEXT_DELTA + GENERATION_COMPLETED live.    |
| **Cancellation (mid-flight)**              | PASS                                 | **PROVEN**                       | Real cancellation between token steps observed.                                                                        |
| **Cancellation (before first token)**      | PASS                                 | **PROVEN**                       | Real test.                                                                                                             |
| **Session isolation**                      | PASS                                 | **PROVEN**                       | Real concurrent-session test.                                                                                          |
| **Runtime restart**                        | recovery after failed load           | **PARTIALLY_PROVEN**             | Only one-cycle recovery from a MANIFEST-MISSING failure. Ten application-restart cycles were not run.                  |
| **Runtime crash recovery**                 | (not claimed)                        | **NOT_RUN**                      | The in-process design has no supervised subprocess to crash. §6 requires either a real subprocess supervisor or executable evidence that in-process is safe under crash. Neither exists yet. |
| **Network denial**                         | static AST scan passes               | **ASSERTED_ONLY**                | No dynamic socket-level or OS-level denial run was executed. §8 explicitly requires this.                              |
| **Sequential-request soak**                | 25 requests complete                 | **PARTIALLY_PROVEN**             | Real 25-request soak (~2 min). Not §9's 3600-second soak.                                                              |
| **One-hour soak**                          | (not claimed)                        | **NOT_RUN**                      |                                                                                                                        |
| **Memory trend**                           | RSS bounded <3000 MB                 | **PARTIALLY_PROVEN**             | Peak checked; monotonic-growth trend NOT sampled at intervals.                                                         |
| **Application restart cycles**             | (not claimed)                        | **NOT_RUN**                      |                                                                                                                        |
| **Process orphan detection**               | in-process, no children              | **NOT_APPLICABLE_UNTIL_R3**      | Under the current in-process design, orphan-detection is trivially zero. Under a real supervised-subprocess design (R3), it must be tested.                                                    |
| **Frozen-runtime execution**               | spec updated                         | **NOT_RUN**                      | No frozen build has been produced.                                                                                     |
| **Installer build**                        | (not claimed)                        | **NOT_RUN**                      |                                                                                                                        |
| **Clean install**                          | (not claimed)                        | **NOT_RUN**                      |                                                                                                                        |
| **Installed generation**                   | (not claimed)                        | **NOT_RUN**                      |                                                                                                                        |
| **Uninstall**                              | (not claimed)                        | **NOT_RUN**                      |                                                                                                                        |

---

## 3. Ladder recalibration

The prior status `FUNCTIONAL_UNPACKAGED_DESKTOP` implied that the
desktop runs "unpackaged" — i.e. as an installable-but-not-installed
application on a machine that isn't the developer's checkout. Nothing
at `d91a836` proves that: every test consumes the source-tree
`release-assets/aeon-desktop-p2-proxy/` bundle from the repository
itself. Frozen-runtime execution outside the repository has never been
attempted.

The honest floor is **`FUNCTIONAL_LOCAL_BUILD`**: real dynamic tests
pass on the developer's machine against the source-tree release
bundle; the export equivalence, the streaming path, session isolation,
Stop, shutdown, and the 25-request-soak subset all pass live. That is
the ladder position the R1–R5 tranches will build upon before any
Windows-side claim can be made.

---

## 4. What R0 does NOT change

* No production code modified.
* No test modified.
* Regression at `d91a836` remains 59 files, 673 checks, 0 failing.
* No research claim is renegotiated — the L-series claim reconciliation
  at `377914b` (Level 2 CANDIDATE_NOT_CLOSED) still stands.
* ACIS default remains OFF.
