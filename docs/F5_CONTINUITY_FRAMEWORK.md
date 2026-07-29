# F5 — Defensive Continuity Framework

**Source of truth:** `aeon/continuity.py`. **Enforcement:** `tests/test_continuity.py` (16 checks).

Generic, bounded continuity mechanisms for authorised manufacturing and communications environments. **Aeon is not connected to real protected equipment** by this upgrade, and Aeon is **not granted direct operational authority** for consequential actions.

## 1. State machine (§F5.1)

Seven deterministic states:

`NORMAL` → `ELEVATED_OBSERVATION` → `DEGRADED` → `CONTAINMENT` → `RECOVERY_PENDING` → `RECOVERING` → `NORMAL`, with `SAFE_HALT` reachable from every state on essential-guarantee loss.

Transition table: `aeon/continuity.py::TRANSITION_TABLE`. Every transition declares:

- **Trigger class** (e.g. `anomaly_observed`, `integrity_failure`, `operator_recovery_authorized`).
- **Permitted initiator** (`aeon_analytical` / `aeon_or_operator` / `operator`).
- **Required deterministic authorization** — consequential transitions (into `CONTAINMENT`, `RECOVERING`, `SAFE_HALT`) require an `operator_authorization_ref` **except** for two safety cases:
  - Aeon MAY autonomously enter `CONTAINMENT` on `integrity_failure` (fail-closed §F4.6).
  - Aeon MAY autonomously enter `SAFE_HALT` on `essential_guarantee_lost` (fail-closed §F4.6).
- **Allowed / prohibited actions** per state (`ALLOWED_ACTIONS`, `PROHIBITED_ACTIONS`).
- **Audit event** (recorded by `ContinuityController._emit_audit`).

Aeon MAY recommend transitions. Aeon MAY NOT unilaterally authorize consequential *positive* transitions (e.g. exiting `SAFE_HALT`, starting a recovery). Verified by `test_aeon_cannot_unilaterally_authorize_recovery` and `test_safe_halt_exit_requires_operator`.

## 2. Manufacturing continuity abstractions (§F5.2)

`analyze_manufacturing_telemetry(frames, ...)` returns a list of `ManufacturingObservation`s. Each observation carries `kind`, `severity`, structured `evidence`, `confidence`, and a `missing_data` flag.

Analytical checks implemented:

- missing / stale telemetry (per-source age vs. `staleness_seconds`)
- sensor disagreement (spread across sensors reporting the same signal)
- process drift (z-score against a per-frame baseline)
- quality-control deviation
- dependency / material interruption
- schedule degradation
- confidence and missing-data status

Not implemented in this phase (and out of Aeon's scope): vendor-specific machinery commands, direct control protocols, actuator drivers. Outputs are always recommendations or structured observations.

## 3. Communications continuity abstractions (§F5.3)

`analyze_comms_telemetry(frames, ...)` returns `CommunicationsObservation`s covering:

- link availability
- message freshness (age)
- authentication status
- sequence continuity + duplicate/replay indicators
- latency changes (vs. baseline)
- bandwidth pressure, queue growth, routing degradation

**Explicitly not present** in Aeon's continuity module: unauthorized interception, jamming, exploitation, credential capture, covert access, unapproved protocol manipulation. Every function is analytical.

## 4. Graceful-degradation order (§F5.4)

`DEGRADATION_ORDER` encodes the mandated order and is unit-tested to keep every `preserve_*` step BEFORE any `reduce_*` step, with `safe_halt_when_essentials_cannot_hold` last:

1. preserve human-safety boundaries
2. preserve certificate and integrity checks
3. preserve artifact authentication
4. preserve state and checkpoint integrity
5. preserve critical telemetry validation
6. preserve essential anomaly detection
7. reduce optional diagnostics
8. reduce nonessential generation
9. reduce context / batch within validated limits
10. SAFE_HALT

Security and integrity controls must not be disabled first. The invariant is enforced by `test_degradation_order_preserves_security_first`.

## 5. Simulation harness (§F5.5)

Synthetic fixtures live in `aeon/continuity.py` alongside the analyzers:

- `synthetic_normal_manufacturing_frames`
- `synthetic_stale_frame`
- `synthetic_normal_comms_frames`

Additional scenarios (missing / conflicting / auth-failure / recovery / safe-halt) are constructed inline in the tests. No restricted or real operational data is used or committed.

## 6. Exit gate

- [x] State transitions are deterministic (`TRANSITION_TABLE`).
- [x] Aeon cannot grant itself consequential authority beyond fail-closed safety (tests).
- [x] Manufacturing and communications capabilities remain analytical.
- [x] Graceful degradation follows the declared order.
- [x] Synthetic scenarios exercise the state machine and every analyzer path.
- [x] No offensive functionality present (source scan + explicit exclusion list in the doc).
- [x] Suite: 101 inherited + 16 F5 = **117/117 pass.**
