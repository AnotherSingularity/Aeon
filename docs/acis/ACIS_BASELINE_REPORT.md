# ACIS Baseline Report

**Base commit:** `ee28f48`
**Ledger head at closure:** ACIS-8 (this tranche)
**Regression at closure:** 616 explicit checks, 0 failures.

---

## 1. Transport Overhead — OFF vs BUCKET

ACIS is measured by two independent surfaces:

* **Structural surface** (this repo): every module in
  `aeon/shuttle/` is exercised by an executable test suite
  that returns non-zero on any refusal path.
* **Runtime surface** (calibration): `aeon/shuttle/calibration.py`
  aggregates `TransportSample` records taken from a live
  BUCKET-mode boundary trace. On this branch, we have not yet
  produced a workload-representative BUCKET calibration
  measurement — that step requires a corpus-authorized
  training run.

Because a full BUCKET calibration report requires a live
recursion trace against a corpus (blocked at STATE B), this
baseline report **does not certify a specific runtime overhead
number**. It certifies the shape of the measurement: what is
sampled, what is included in `transport_ms`, and what would
trigger refusal.

Baseline current-transport measurement (structural, taken by
inspecting `HybridModel.forward` at `ee28f48`):

| Field                                        | Value at OFF |
| -------------------------------------------- | ------------ |
| Existing broadcasts per K-boundary           | 1            |
| Existing clones per K-boundary               | 0            |
| Existing copy volume in default path         | 0            |
| Device-transfer volume (single-device run)   | 0            |
| Duplicate-state lifetime                     | none         |
| Byte-identity of OFF-mode forward            | preserved    |

Byte-identity is enforced by the shuttle-optional guard
`if shuttle is not None:` in `HybridModel.forward` and the AST
test `test_hybrid_shuttle_import_is_guarded_by_none_check` in
`test_acis_3_shuttle.py`.

---

## 2. Calibration Methodology

`aeon/shuttle/calibration.py` defines the certified sampling
API. A single `TransportSample` covers:

* `forward_ms` — wall-clock of the surrounding forward pass.
* `transport_ms` — wall-clock added by shuttle bookkeeping:
  event allocation, digest computation, lane state, coherence
  assertion. Excludes model math, autograd, and gradient
  computation.
* `time_to_first_broadcast_ms` — from boundary start to the
  first successful `publish_broadcast` call.

Overhead is
`mean(transport_ms) / mean(forward_ms)`.

The BUCKET certification gate is:

    overhead ≤ 0.01  (transport ≤ 1% of forward)

CONVEYOR is refused unless BUCKET is itself certifiable AND
conveyor overhead ≤ bucket overhead AND semantic identity AND
autograd identity are both preserved on conveyor mode.

---

## 3. Refusal Surface

Every calibration and rollout refusal is named and tested:

| Code                              | Test                                                        |
| --------------------------------- | ----------------------------------------------------------- |
| `no_samples`                      | `test_summarize_refuses_no_samples`                         |
| `bucket_overhead_too_high`        | `test_conveyor_refused_when_bucket_over_budget`             |
| `no_conveyor_evidence`            | `test_conveyor_refused_when_no_conveyor_evidence`           |
| `conveyor_slower_than_bucket`     | `test_conveyor_refused_when_slower`                         |
| `conveyor_semantic_divergence`    | `test_conveyor_refused_when_semantic_divergence`            |
| `conveyor_autograd_divergence`    | `test_conveyor_refused_when_autograd_divergence`            |
| `all_gates_passed` (certified)    | `test_conveyor_certified_only_when_all_gates_pass`          |

The certified default state at ACIS-8 closure:

* `bucket_certified`: **structurally certifiable pending live
  workload measurement** — the STRUCTURAL invariants required
  for BUCKET certification are all locked. A runtime overhead
  number will be filled in by the calibration harness the
  first time BUCKET mode is enabled against a corpus-authorized
  training run.
* `conveyor_certified`: **false** — no measured evidence exists.
  Under the certified default, `decide_conveyor` returns
  `conveyor_refused / no_conveyor_evidence`.

---

## 4. What This Report Does NOT Claim

* It does NOT claim BUCKET mode has been benchmarked on a real
  training loop. That step waits on the corpus authorization at
  `AWAITING_OFFLINE_CORPUS_SOURCES`.
* It does NOT claim CONVEYOR_EXPERIMENTAL is safe. The certified
  default is `conveyor_refused`.
* It does NOT claim K has ever changed. K is `16`, fixed, and
  no code path — not even backpressure — can alter it.
