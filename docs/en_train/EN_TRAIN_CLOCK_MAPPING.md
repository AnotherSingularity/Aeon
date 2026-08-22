# EN-TRAIN — Aeon Dual-Clock Repository Mapping

**Produced before** the corrected mathematical spec assigns any meaning
to `C_1` or `C_2`. Sourced entirely from repository evidence at
`447f0dc`.

**Two clocks live in the repository:** the **FAST CLOCK** (per-token
substrate tick inside each K-window) and the **SLOW CLOCK** (once-per-
K-window recursion tick + one shared broadcast + transformer inject).
No formal `C_1` / `C_2` symbol appears anywhere in the source tree —
the corrected spec will introduce those as bookkeeping labels bound to
these two clocks in the same order.

Machine-readable: `docs/en_train/en_train_clock_mapping.json`.

The English-training tranche **does not modify** either clock, their
coupling, K=16, the σ<margin certificate, the substrate port, or the
Recursion joiner's `.step` semantics. The EN-TRAIN infrastructure at
`447f0dc` calls `HybridModel.forward` unchanged and consumes its
logits only through the loss.

---

## 1. Canonical names (in repo)

* **"fast clock"** — `aeon/hybrid.py:11` (docstring:
  *"fast clock (per token, within a K-window w; conditioning state = h_{w-1})"*)
* **"slow clock"** — `aeon/hybrid.py:14`, also `aeon/__init__.py:18`,
  `aeon/shuttle/__init__.py:7`, `aeon/shuttle/__init__.py:28`,
  `aeon/shuttle/backpressure.py:4`, `aeon/recursion.py:35`,
  `docs/TOPOLOGY_MAP.md:35`, `docs/PRESERVATION_MANIFEST.md:21` (P-K16 row)

The names **"fast clock"** and **"slow clock"** are the canonical
repository terms.

---

## 2. Fast clock

| Property                         | Value                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------- |
| Cadence                          | exactly **once per input token** index `i ∈ [start, end)` inside each K-window                    |
| Update trigger                   | `for i in range(start, end):` at `aeon/hybrid.py:154-158`                                          |
| Call site                        | `r_i = self.substrate.step(x_i)` at `aeon/hybrid.py:156`                                          |
| Prepared by                      | `self.substrate.reset(B, device)` at `aeon/hybrid.py:141`                                         |
| Boundary detach                  | `self.substrate.detach_state()` at `aeon/hybrid.py:178` (design note D2)                          |
| State variable                   | substrate cell state (matrix `S ∈ ℝ^{H×N×N}` in `MatrixCell` / vector in `VectorCell`)            |
| Concrete cell impls              | `aeon/substrate/matrix_cell.py::MatrixCell`, `aeon/substrate/vector_cell.py::VectorCell`         |
| Port contract                    | `aeon/substrate/port.py::SubstratePort`                                                            |
| Interaction w/ slow clock (in)   | `x_i = emb_proj(emb_i) + cond_proj(h_{w-1})` at `aeon/hybrid.py:151, 155` — fast tick READS `h_{w-1}` |
| Interaction w/ slow clock (out)  | `s_w = s_proj(mean_i r_i)` at `aeon/hybrid.py:160-161` — window-mean of fast readouts feeds the slow tick |
| Interaction w/ transformer stream | fast tick uses `emb_i` from `transformer.embeddings(input_ids)` at `aeon/hybrid.py:134`             |
| Configuration                    | no direct knob — cadence is one tick per input token by construction                              |

**Fast-clock tests (indirect):**
* `tests/test_substrate_port.py` — port shape / dtype contract
* `tests/test_recursion_topology.py::test_recursion_step_called_once_per_window` — proves the slow clock is NOT per-token

---

## 3. Slow clock

| Property                         | Value                                                                                             |
| -------------------------------- | ------------------------------------------------------------------------------------------------- |
| Cadence                          | exactly **once per K-window** `w`, `num_windows = ceil(T / K)`                                    |
| Update trigger                   | `for w in range(num_windows):` at `aeon/hybrid.py:146`                                            |
| Call site                        | `h, c = self.recursion.step(s_w.float(), t_w.float(), h.detach(), c.detach(), e=e_w.float() if e_w else None)` at `aeon/hybrid.py:175-177` |
| Prepared by                      | `h, c = self.recursion.init_state(B, device=device)` at `aeon/hybrid.py:142`                       |
| Shared broadcast (per window)    | `h_cond = h` at `aeon/hybrid.py:150`; broadcast to each fast token via `inject_cols.append(h_cond)` at 158; written to hidden at end of forward via `transformer.inject(...)` |
| State variables                  | `h ∈ ℝ^{H_rec}` (fp32), `c ∈ ℝ^{H_rec}` (fp32) — carried between windows                          |
| Update rule                      | `aeon/recursion.py:148-169` — `h_next = tanh(W_s·s + W_t·t [+ W_e·e] + h·W_hᵀ + c_next)`, `c_next = (1−λ)c + λ tanh(hWc^T)` |
| σ certificate constants          | `MARGIN_H = model.margin_h`, `MARGIN_C = model.margin_c` (`aeon/recursion.py:100-101`, aeon/recursion.py:22-23) |
| Interval `K`                     | **16** at `aeon/hybrid.py:68` (`K: int = 16` default) and `:78` (`self.K = K`); pinned by every config: `configs/latent_bypass/aeon_lbc1_proxy.yaml:20`, `configs/aeon_v1.yaml:6`, `configs/aeon_350m.yaml:14`; also `aeon/shuttle/__init__.py:29` (`FIXED_K: int = 16`) |
| Interval immutability            | preserved per `docs/PRESERVATION_MANIFEST.md:21` (P-K16 row); asserted by `tests/test_recursion_topology.py::test_K_is_16_and_not_adaptive` |
| Interaction w/ fast clock        | receives `s_w = s_proj(mean_i r_i)` from fast readouts; detaches substrate at boundary (`aeon/hybrid.py:178`) |
| Interaction w/ transformer       | picks `t_w = t_all[:, end - 1, :]` at `aeon/hybrid.py:162`; writes back via `transformer.inject(hidden, inject_signal)` |

**Slow-clock tests:**
* `tests/test_recursion_topology.py::test_K_is_16_and_not_adaptive`
* `tests/test_recursion_topology.py::test_recursion_stays_fp32_after_cast`
* `tests/test_recursion_topology.py::test_certificate_holds_by_construction`
* `tests/test_recursion_topology.py::test_certificate_fails_closed_on_forced_violation`
* `tests/test_recursion_topology.py::test_single_broadcast_shared_source`
* `tests/test_recursion_topology.py::test_recursion_step_called_once_per_window`

---

## 4. Coupling between the two clocks

* **Fast conditions on slow:** `h_{w-1}` conditions every fast tick within window `w`
  via `cond_proj` (`aeon/hybrid.py:150-155`).
* **Fast feeds slow via aggregation:** `mean_i r_i → s_proj → s_w` (`:160-161`).
* **Slow-clock boundary detach:** `h.detach()`, `c.detach()`, and
  `substrate.detach_state()` at each window end — truncated BPTT (D2).
* **Slow writes transformer:** `logits = lm_head(hidden + γ · write_proj(inject_signal))`
  where `inject_signal[window w tokens] = h_{w-1}` (docstring lines 22-24).

---

## 5. Documentation-vs-runtime discrepancies found

* **`docs/TOPOLOGY_MAP.md` line-number citations for the window loop are STALE.**
  Doc says "the `for w in range(num_windows)` loop in `hybrid.py:127-151`
  calls `self.recursion.step(...)` once per window". Runtime: the loop is at
  `aeon/hybrid.py:146-178`. **Content match, line numbers drifted.**
  Severity: documentation-only drift. No semantic conflict; the runtime
  cadence still runs the loop exactly once per K-window and
  `tests/test_recursion_topology.py::test_recursion_step_called_once_per_window`
  confirms it.

* **`docs/TOPOLOGY_MAP.md` cites `hybrid.py:139` (inject_cols.append)
  and `hybrid.py:154` (transformer.inject).** Runtime: `.append(h_cond)`
  is at `aeon/hybrid.py:158`; the inject call is later in the forward.
  Same class of drift.

**Neither drift is repaired in this tranche** — the correction order
explicitly says *"Do not modify C_1, C_2, their coupling, or their
update semantics during the English-training tranche."* Repairing
canonical clock references is deferred out of caution and left as
follow-up documentation work.

---

## 6. Modules intentionally frozen by existing design

Repository search: **no explicit `requires_grad_(False)` calls** in
`aeon/hybrid.py`, `aeon/recursion.py`, or `aeon/substrate/*`.
Every trainable parameter is trainable by default at construction.
`.detach()` is used **locally, per-forward, for truncated BPTT** at
window boundaries — that is an autograd-graph choice, not a permanent
freeze.

There is one **operator-selectable freeze**:
`HybridModel(freeze_backbone: bool = False, ...)` at `aeon/hybrid.py:73`
— defaults to **False**, and `docs/training/p2_evidence.json` records
that the P2 checkpoint was trained with the default.

No exemption is claimed.

---

## 7. Summary — safe input to the corrected mathematical spec

Two clocks exist in the repository:

* **fast clock** — per-token substrate tick, aeon/hybrid.py:154-158
* **slow clock** — per-K-window recursion tick + shared broadcast +
  transformer inject, aeon/hybrid.py:146-178 + aeon/recursion.py:148

The corrected mathematical spec at
`docs/en_train/EN_TRAIN_CORRECTED_MATHEMATICAL_SPEC.md` binds:

* `C_1 := fast clock` (as above)
* `C_2 := slow clock` (as above)

purely as **bookkeeping labels for the mapping**. No new equation
changes the runtime semantics of either clock.
