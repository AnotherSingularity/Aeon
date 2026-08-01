# ACIS — Coherent Information Shuttling Architecture

**Status:** Certified — V0.02 Architecture Preserved
**Base commit:** `ee28f48` (STATE B stopping point)
**Final commit:** ACIS-8 (this tranche)
**Certified default mode:** `OFF`

---

## 1. Purpose

ACIS ("Aeon Coherent Information Shuttling") is a transport and lifecycle
layer that sits BENEATH Aeon's cognition. It provides:

* A single, immutable Recursion broadcast per K-boundary.
* Two read-only leases (Transformer, Substrate) that resolve to the
  SAME live tensor object.
* Explicit ownership transfer for mutable state capsules.
* Bucket-brigade lane discipline with capacity + freshness + backpressure.
* Recovery, replay, quarantine, and coherence enforcement.
* A calibration gate before any optional CONVEYOR_EXPERIMENTAL mode.

ACIS **never** changes:

* Transformer, substrate, or Recursion semantics.
* Recursion slow-clock interval `K = 16`.
* Model parameters, training objective, tokenizer, or corpus path.
* Substrate gate inputs.
* Stream isolation (no direct transformer↔substrate calls).
* Single-broadcast semantics per boundary.
* Autograd graph identity of the payload tensor.

---

## 2. Rollout Modes

Defined in `aeon/shuttle/policy.py`.

| Mode                     | Behavior                                                                     |
| ------------------------ | ---------------------------------------------------------------------------- |
| `OFF` (default)          | `HybridModel.forward` runs unchanged. Zero ACIS code executes.               |
| `OBSERVE`                | Broadcasts + leases + events recorded, but no transport pre-registration.    |
| `BUCKET`                 | Full bucket-brigade lane discipline. Certified when transport overhead ≤ 1%. |
| `CONVEYOR_EXPERIMENTAL`  | Lane pre-registration. REFUSED by default until measured evidence certifies. |

The parser fails closed on unknown modes.

---

## 3. Module Map

Every module lives under `aeon/shuttle/`.

| Module           | Responsibility                                                     |
| ---------------- | ------------------------------------------------------------------ |
| `policy.py`      | `ShuttleMode` enum + parser + `is_default_off`.                    |
| `audit.py`       | `AcisEvent` (frozen, no payload/tensor/bytes) + chained ledger.    |
| `contracts.py`   | `RepresentationContract` + basic validation + zone constants.      |
| `broadcast.py`   | `ImmutableRecursionBroadcast` + `compute_semantic_digest`.         |
| `lease.py`       | `BroadcastLease` + delivery-state machine + resolve.               |
| `lifecycle.py`   | `BroadcastCustody` — atomic issue_pair / revoke / retire.          |
| `capsule.py`     | `MutableStateCapsule` + 7 nominal + 5 failure states.              |
| `ownership.py`   | `OwnershipLedger` — chained digest + single-owner enforcement.     |
| `lane.py`        | `BucketLane` — capacity, FIFO pop, duplicate suppression, cancel.  |
| `freshness.py`   | `FreshnessPolicy` + rejects stale/future/dup/causal_mismatch/exp.  |
| `backpressure.py`| Transport-only admission bound; refuses altering K.                |
| `quarantine.py`  | `QuarantineRegistry` — reason + evidence + audit history.          |
| `coherence.py`   | `assert_pair_coherent` — same broadcast, digest, and live tensor.  |
| `replay.py`      | `ReplayJournal` — refuses regression + pre-stop replay.            |
| `recovery.py`    | `RecoveryController` — Safe Stop → drain → checkpoint → resume.    |
| `calibration.py` | Transport measurement + BUCKET/CONVEYOR decision gates.            |
| `routing.py`     | `AcisBoundaryShuttle` protocol + `StandardAcisShuttle`.            |

---

## 4. Boundary Flow (BUCKET mode)

At every K-boundary:

1. `publish_broadcast(payload=h_cond, contract=…)` produces an
   `ImmutableRecursionBroadcast`.
2. `BroadcastCustody.issue_pair(broadcast)` atomically issues two
   `BroadcastLease` objects — one for Transformer, one for Substrate.
3. Each consumer calls `resolve_lease` which returns
   `broadcast.payload_handle` — the SAME live tensor object.
4. `assert_pair_coherent` verifies both leases point at the same
   broadcast, same semantic digest, and same Python object.
5. Consumers acknowledge, custody retires the broadcast, and the
   lane FIFO-pops the destination slot.

`h_cond` flows into `inject_cols` UNCHANGED. The shuttle's semantic
digest computation uses `detach().to(float32).contiguous()` for
hashing only — the payload's autograd connection is not altered.

---

## 5. Invariants

Certified and locked by the test matrix:

| Invariant                                                | Enforced by                          |
| -------------------------------------------------------- | ------------------------------------ |
| `K = 16` fixed, no adaptive-K component                  | `contracts.validate_contract_basic`  |
| One broadcast per K-boundary                             | `broadcast.publish_broadcast`        |
| Both leases resolve to same semantic identity            | `coherence.assert_pair_coherent`     |
| Recursion state fp32                                     | `contracts.RepresentationContract`   |
| Substrate autonomous (no direct transformer↔substrate)   | `stream_independence` test           |
| Six V0.02.02 patches intact                              | `test_six_patches.py`                |
| OFF-mode bit-identical to shuttle-absent build           | `test_acis_3_shuttle`                |
| No outbound network                                      | `runtime_policy` test                |
| Immutable broadcast + read-only leases                   | `broadcast` / `lease`                |
| Single mutable ownership                                 | `ownership.enforce_single_mutable_*` |
| No adaptive-K assertion                                  | tests never assert `iters == 16`     |
| Backpressure never alters K                              | `backpressure.assert_no_cognition_*` |

---

## 6. Failure Modes

Every failure mode is a NAMED refusal with a code and detail. Nothing
"silently degrades." Named refusals:

* `capacity_exceeded`, `duplicate_capsule`, `invalid_lane_stage_transition`
* `stale_source_epoch`, `future_source_epoch`, `duplicate_admission`,
  `superseded`, `causal_mismatch`, `expired`
* `epoch_regressed`, `cognition_side_effect`
* `broadcast_id_divergence`, `semantic_digest_divergence`,
  `payload_identity_divergence`
* `readmission_refused`, `already_quarantined`
* `journal_closed`, `boundary_replay_refused`
* `drain_without_stop`, `lane_not_drained`, `resume_without_stop`
* `bucket_overhead_too_high`, `no_conveyor_evidence`,
  `conveyor_slower_than_bucket`, `conveyor_semantic_divergence`,
  `conveyor_autograd_divergence`

---

## 7. What ACIS Does NOT Change

For emphasis:

* No new tokens, layers, losses, or gradients.
* No routing/token change.
* No transformer↔substrate direct wire.
* No Recursion cadence change.
* No adaptive-K control capsule.
* No writable broadcasts.
* No shared mutable state.
* No external network egress.
* No modification to the six V0.02.02 patches.

---

## 8. Certification Evidence

See:

* `docs/acis/ACIS_BASELINE_REPORT.md` — measurement methodology.
* `docs/acis/ACIS_CERTIFICATION_REPORT.md` — pass/refuse summary.
* `docs/acis/acis_final_evidence.json` — machine-readable evidence.
* `docs/acis/acis_status.json` — machine-readable ledger status.
