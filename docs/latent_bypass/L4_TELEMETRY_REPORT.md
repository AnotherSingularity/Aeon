# L4 Bypass Telemetry Report

**Fixed experimental basis:** `runs/aeon_lbc1_P2/final.pt`
**Development partition:** CALIBRATION (PG-84, 821 records)
**Confirmatory partition:** VALIDATION (PG-55, 1164 records)
**Sealed partition:** TEST (PG-1661) — untouched by L4.
**Runner:** `scripts/run_l3_l4_l5.py`.

Telemetry is sampled, detached, byte-bounded, local, and disabled by
default in the certified default forward. It runs here only under an
explicit script invocation; nothing about the model or the ACIS
default OFF policy changes.

---

## 1. Per-boundary telemetry means

Captured via `RecursionJoiner.step`-wrap in `torch.no_grad()`. Each
boundary contributes an 8-vector (batch) of Recursion state and its
source signals.

|                                | CALIBRATION |  VALIDATION |
| ------------------------------ | -----------:| -----------:|
| K-boundaries captured          |         512 |         416 |
| mean ‖h_w‖                     |     ~variable — see JSON |
| mean ‖h_new − h_prev‖ (Δ)     |     ~variable — see JSON |
| mean ‖t_w‖ (transformer src)   |     ~variable — see JSON |
| mean ‖s_w‖ (substrate src)     |     ~variable — see JSON |
| mean z_norm (reaction coord)   |     ~variable — see JSON |
| mean z_dir  (reaction coord)   |     ~variable — see JSON |
| mean batch loss                |     ~variable — see JSON |

Exact numeric means are in `docs/latent_bypass/l4_telemetry_evidence.json`.
The CALIBRATION and VALIDATION means differ modestly (typical of a
model at ~6 mean loss on both partitions) — no distributional
divergence red flag.

---

## 2. Sampling policy

* Sample rate: every K-boundary in each of 32 batches per partition.
* Byte bound: per-boundary snapshot is ≤ (4 × float32 × 128 × 8) ≈ 16 KiB.
* Persistence: none — every capture ends when the script exits.
* Instrumented-vs-uninstrumented overhead: the wrap adds one
  ``.detach().cpu().float().numpy()`` per boundary; measured empirically
  the L3/L4 pass finished in ~15 s per partition (bounded by the
  small model's forward wall-clock, not by capture).

The certified default forward does NOT run this telemetry. It fires only
under an explicit `scripts/run_l3_l4_l5.py` invocation.

---

## 3. Pre/post-broadcast loss — deferred to L5

Full per-boundary Δℓ_b = ℓ_pre − ℓ_post (broadcast-zero counterfactual)
requires an in-forward hook and a paired counterfactual for every
boundary — that is exactly what L5's ZERO_BROADCAST intervention
delivers, at the partition level. This report therefore reports the
mean-loss surface and defers the boundary-level Δℓ analysis to L5's
paired causal comparison, which lives in `L5_CAUSAL_REPORT.md`.

---

## 4. Claim level supported

* **Level 1 — STRUCTURALLY_IMPLEMENTED** on the L4 surface alone.
* Level 2 (OBSERVATIONAL) requires the L3 calibration lock + a
  confirmatory VALIDATION analysis that we describe here but do
  not elevate to a positive scientific claim until it is
  paired with L5's held-out causal signal on the sealed test.

Full evidence: `docs/latent_bypass/l4_telemetry_evidence.json`.
