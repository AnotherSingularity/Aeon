# Aeon V0.02.03 — Topology Map

Machine-readable form: `docs/topology.json`. This document is the human-readable partner.

## Two independent streams

### Substrate stream

- **Entry point:** `aeon/hybrid.py::HybridModel.forward` calls `self.substrate.step(x_i)` per token inside each K-window.
- **Cell:** built by `aeon/substrate/__init__.py::make_substrate` from config `kind: matrix|vector`.
  - `MatrixStateCell` (`aeon/substrate/matrix_cell.py`): `S ∈ (B, H, N, N)` matrix state; per-channel decay `w = sigmoid(decay_logit)`; recurrence `S_t = k⊗v + w·S_{t-1}`; readout `tanh(readout(r·S))` bounded in (-1,1); optional adaptive feedback (§below).
  - `VectorStateCell` (`aeon/substrate/vector_cell.py`): single vector state, `tanh(W_x·x + scalar·W_h·h)`. Deliberately simple (see `_vector_simplicity` conformance).
- **Per-cell state dtype:** follows parameter dtype (`matrix_cell.py:98`, `vector_cell.py:53` — patch 4e).
- **Substrate → Recursion input:** `s_w = s_proj(mean_i r_i)` in `hybrid.py:141-142`. `s_proj: d_state → h_rec`.
- **Broadcast consumption:** `cond_in = cond_proj(h_{w-1})` in `hybrid.py:132`. `h_{w-1}` is the previous window's Recursion state, cast to compute dtype. Substrate input for each token in window w is `emb_in[t] + cond_in` — the substrate never reads any transformer state.
- **Feedback controller** (`aeon/substrate/feedback.py`, applied only in `matrix_cell`): reads only its own prior readout — `L(t) = EWMA(|base_t − base_{t-1}|)` — computes `g(L) = sigmoid(α(L−θ))`, blends `(1−g)·base + g·output_bound·tanh(W_stressed·base)`. Autonomous per directive §3.6.

### Transformer stream

- **Entry point:** `aeon/hybrid.py::HybridModel.forward` calls `self.transformer.hidden_states(input_ids=…)` once per forward.
- **Model:** `aeon/transformer.py::AeonTransformer` — RMSNorm, GQA (16 heads, 4 KV heads), SwiGLU MLP, tied lm_head. Rotary `inv_freq` computed fresh fp32 per forward (patch 4f).
- **Transformer → Recursion input:** `t_all = self.transformer.read(hidden)` in `hybrid.py:119`; then `t_w = t_all[:, end-1, :]` (last-token readout per window) in `hybrid.py:143`. `read_proj: D → h_rec`.
- **Broadcast consumption:** `inject_signal[window w tokens] = h_{w-1}` (broadcast held across the window) built in `hybrid.py:139` and applied by `HybridTransformer.inject(hidden, signal)` in `transformer.py:263-266`: `(hidden.float() + γ · write_proj(signal).float()).to(dtype)` — patches 4c + 4d + 4b + 4a.

## Recursion: the integration point

- **File:** `aeon/recursion.py::RecursionJoiner`.
- **Inputs:** `s, t` (both `(B, h_rec)`), optional `e = mean_i emb_i` (embedding side-input, projected internally by `W_e`), and `h, c` (the previous carried state, `.detach()`ed at the slow-clock boundary for truncated BPTT — `hybrid.py:149`).
- **Update:** `h = tanh(W_s·s + W_t·t + W_e·e + W_h·h + c)`; carry `c = λ·c + (1−λ)·h` with `λ = 0.5`.
- **Certificate — structural by construction:**
  `W = sigmoid(s)·MARGIN·Cayley(A)·diag(tanh(d))` yields `σ(W) < MARGIN` for both `W_h` (MARGIN_H=0.98) and `W_c` (MARGIN_C=0.95). `audit()` returns `sigma_Wh`, `sigma_Wc`, `holds`.
- **Precision:** the Recursion module is kept fp32 by `model.recursion.float()` (`scripts/train.py:116`, `scripts/infer.py:43`) — required by the Cayley solve / SVD in the certificate audit and by the σ-bound guarantees.
- **Cadence:** exactly once per slow-clock window. The `for w in range(num_windows)` loop in `hybrid.py:127-151` calls `self.recursion.step(...)` once per window — never per token.

## Slow clock

- **K = 16**, defined in `aeon/hybrid.py:68` (`K: int = 16`) and set in both configs (`configs/aeon_350m.yaml:14`, `configs/aeon_v1.yaml:6`).
- Not adaptive. Not entropy-triggered. Not learned. The E1 configuration-invariant tests will assert this in code.

## Single broadcast

- The joiner emits **one** `h_w` per window.
- Both streams consume the **same** `h_{w-1}`:
  - **substrate:** through `cond_proj(h_{w-1})` added to every token's substrate input in window w (`hybrid.py:132, 136`).
  - **transformer:** through `inject(hidden, inject_signal)` at the end of the forward (`hybrid.py:154`), where `inject_signal` is `h_{w-1}` broadcast per-token within each window (`hybrid.py:139`, then stacked at 153).
- There is **no** `J_S(r_b)` / `J_T(r_b)` split — same tensor, two consumers. The E1 topology test asserts identity of the broadcast source.

## Entry points

- **Training:** `scripts/train.py::main` — YAML-driven. Loads Aeon tokenizer + corpus when configured; else synthetic tokens.
- **Inference:** `scripts/infer.py::main` — YAML-driven; greedy.
- **Tokenizer training:** `scripts/train_tokenizer.py::main` (importable `train_tokenizer()`).
- **Diagnostics (feedback):** `scripts/diagnose_feedback.py::main` — runs the five fault-isolation diagnostics on a saved checkpoint. E4 extends this to an offline diagnostic tool with more probes.

## State reset

- Substrate `reset(B, device)` follows param dtype (patch 4e).
- Recursion carry `h, c` initialised via `RecursionJoiner.init_state`; carried across windows within a forward and `.detach()`ed at each window boundary for truncated BPTT.
- Feedback controller `reset()` clears sensor state at each forward.

## Checkpoint save/restore (pre-E3)

- `scripts/train.py::save_checkpoint` — `torch.save({"step", "model", "optim"}, path)`. E3 replaces this with an atomic writer + integrity + wider preserved state.
- Resume path in `main()` — `torch.load(ck, map_location=device)`, then `load_state_dict`. Hardened in E3 to `weights_only=True` + metadata check.
