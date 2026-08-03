# L3 Reaction Coordinate Report

**Fixed experimental basis:** `runs/aeon_lbc1_P2/final.pt`
**Fit partition:** CALIBRATION (PG-84 Frankenstein preprocessed at
`aeon-lbc1-v1`, 821 records, sha256:bf7f722d…)
**Evaluation partition:** VALIDATION (PG-55 Wizard-of-Oz, 1164 records,
sha256:ed75c7a4…)
**Runner:** `scripts/run_l3_l4_l5.py`

Sealed test (PG-1661) is NOT opened until the L3 calibration lock
lands. The lock is written by `write_calibration_lock()` in the
same script — it fires before `do_L5()` and the lock file is
committed alongside this report.

---

## 1. Data captured

Recursion boundary states were captured by wrapping `RecursionJoiner.step`
during `HybridModel.forward` — evaluation only, no gradient flow, no
back-propagation into Aeon, no routing effect, no substrate-gate
effect, no training-loss effect, no ACIS transport effect.

| Partition   | K-boundaries captured |
| ----------- | ---------------------:|
| CALIBRATION | 512                   |
| VALIDATION  | 416                   |

Each boundary yields a 128-dim (h_rec) vector `h_w` (post-recursion),
its previous `h_prev`, and the two source vectors `s_w` and `t_w`.

---

## 2. Coordinates fit

Two coordinates were fit on CALIBRATION alone:

* **z_norm(h)** = ‖h − h̄‖₂, where h̄ = mean(h_new) on CALIBRATION.
  A centered magnitude scalar. No fitting required beyond h̄.
* **z_dir(h)** = vᵀ (h − h̄), where v is ridge-regressed on
  CALIBRATION against the per-boundary target
  y_b = ‖h_new − h_prev‖₂ (Recursion Δ magnitude), regularization λ = 1.

The predictive-diagnostic coordinate z_pred (§14 Ψ_θ) is not fit for
this bounded configuration — z_dir already saturates on the recursion-
delta target (see §3).

Effective rank of centered H_cal = ~110 (out of dim 128). The Recursion
state uses most of its manifold.

---

## 3. Held-out predictive result

**Evaluated on VALIDATION (never touched by fit):**

| Coordinate | R² on VAL predicting ‖Δh‖ | Shuffled-labels R² control |
| ---------- | -------------------------:| --------------------------:|
| z_norm     |  variable per rescaling   | —                          |
| z_dir      |               **0.9079**  |                     0.0318 |

* Signal above shuffled control: **0.876** — a very strong margin.

**Important caveat:** the target ‖Δh‖ is Recursion's own step
magnitude, which is closely coupled to h itself. High R² is expected
and does not by itself prove that the broadcast carries semantic
information about the next token. It proves only that a linear
combination of h reconstructs Recursion's own update norm well —
i.e. the coordinate is real, stable, and non-trivial, but its
predictive scope is limited to "how much Recursion moved" rather
than "did the broadcast improve token prediction."

L5 (§18) is where the paired causal test on token loss actually lives.

---

## 4. Discipline confirmations

* No gradient enters Aeon (`torch.no_grad()` for every forward).
* No coordinate affects routing, Recursion, substrate gate, training
  loss, or ACIS transport.
* Test partition never opened during this L3 fit.
* All fitting on CALIBRATION only.
* Shuffled-labels control run with fixed seed 20260803.

---

## 5. Claim level supported by L3 alone

* **Level 1 — STRUCTURALLY_IMPLEMENTED.**
* L3 alone does not elevate to Level 2+. Level 2 requires L4 telemetry
  + committed calibration lock + held-out evaluation; the lock lands
  next and L5 delivers the paired causal test.

Machine-readable evidence: `docs/latent_bypass/l3_reaction_coordinate_evidence.json`.
