# ACIS-0 — Invariant lock + current-transport baseline

Base commit: `ee28f48` (offline-corpus intake).
Regression at ACIS-0 start: 513/513.
Scope: **scaffold only**. `HybridModel.forward` is not touched.

## Certified invariants locked (§2)

| Invariant | Location | Test |
|-----------|----------|------|
| K=16 fixed | `aeon/hybrid.py` (`K: int = 16`) + `aeon/shuttle/__init__.py` (`FIXED_K = 16`) | `test_fixed_k_declared_at_16` |
| No adaptive-K component | AST/source sweep of `aeon/` + `scripts/` | `test_no_adaptive_k_component_present` |
| One broadcast per boundary | AST check on `HybridModel.forward` (single `inject_cols.append`, single `transformer.inject`) | `test_single_broadcast_per_window` |
| Recursion state fp32 | Slow-clock tick casts `s_w.float()` / `t_w.float()` before `recursion.step` | `test_recursion_still_fp32_in_slow_clock_tick` |
| Substrate autonomous gate | `substrate.step(x_i)` — no transformer entropy/logits/attention | `test_substrate_gate_inputs_unchanged` |
| Default forward unchanged | `hybrid.py` does not import `aeon.shuttle` | `test_hybrid_does_not_import_aeon_shuttle_at_acis_0` |
| ShuttleMode fail-closed | Unknown strings raise `UnknownShuttleMode` | `test_shuttle_mode_unknown_string_fails_closed` |
| No outbound network in `aeon/shuttle/` | Source sweep | `test_shuttle_package_has_no_outbound_network_reference` |

## Current transport measurement (§12)

The existing Recursion-broadcast path (measured by inspection of the
current `HybridModel.forward`):

- Existing broadcast count per window: **1** — the `h_cond`
  reference appended to `inject_cols` exactly once per token in
  `[start:end]` inside the per-window loop.
- Existing boundary schedule: fixed K=16 slow-clock; `num_windows =
  ceil(T / K)`.
- Clone count in default path: **0** — no `.clone()` on `h_cond`.
  The same tensor reference is appended `end - start` times.
- Copy volume in default path: **0** — `torch.stack(inject_cols,
  dim=1).to(compute_dtype)` performs the stack once, in-place with
  respect to the same broadcast reference.
- Device-transfer volume: 0 for a single-device run.
- Duplicate-state lifetime: none — a single canonical `h_cond` per
  window is the sole broadcast identity.

## Scaffolding landed at ACIS-0

- `aeon/shuttle/__init__.py` — `FIXED_K = 16`, `SHUTTLE_MODE_DEFAULT =
  "off"`, `BASE_COMMIT_ACIS_0 = "ee28f48"`.
- `aeon/shuttle/policy.py` — `ShuttleMode` enum (OFF / OBSERVE /
  BUCKET / CONVEYOR_EXPERIMENTAL), fail-closed `parse_shuttle_mode`.
- `aeon/shuttle/audit.py` — `AcisAuditLog` chained ledger digest;
  `AcisEvent` frozen dataclass that CANNOT hold a payload reference
  (test-enforced field allowlist).

## What ACIS-0 does NOT touch

- `HybridModel.forward` — unchanged. Grep `aeon/hybrid.py` for
  `aeon.shuttle`: no matches.
- `HybridModel.__init__` — unchanged.
- `aeon.recursion` — unchanged.
- `aeon.substrate` — unchanged.
- Six V0.02.02 corrections — intact (guard test:
  `tests/test_six_patches.py`).

## Next: ACIS-1

Wrap the existing Recursion output in metadata under OBSERVE only.
Introduce `ImmutableRecursionBroadcast` + `RepresentationContract`.
No change to broadcast tensor, injection code, destination
interpretation, Recursion update, or stream behaviour. Prove one
broadcast per boundary, one semantic digest, one canonical
representation, correct source/target epoch, correct shape/dtype,
correct model identity, correct architecture identity, fixed K=16.
