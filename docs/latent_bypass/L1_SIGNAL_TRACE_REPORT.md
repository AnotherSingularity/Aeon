# L1 — Authoritative Signal Trace Report

Base: L0.1 (`1b59b8c`).
Instrumentation module: `aeon/bypass/signal_trace.py`.
Trace points wired in: `aeon/hybrid.py::HybridModel.forward`.
Enforcement tests: `tests/test_l1_signal_trace.py` (14 checks).

## What L1 delivers

L1 proves the actual executed tensors follow Aeon's declared signal
route. It adds an optional `observer=None` keyword to
`HybridModel.forward`; when the argument is None (default), the forward
path runs unchanged — no diagnostic allocation, copy, sync,
serialization, or branch beyond the single `if observer is not None:`
guard per K-boundary. When the argument is a `AeonDiagnosticObserver`,
the observer receives one `RecursionWindowEvent` per K=16 boundary
crossed.

## Traced signal route

For every K=16 window boundary the observer receives a scalar summary of:

1. **Transformer stream → Recursion projection** — `t_w` at `t_all[:, end-1, :]`.
2. **Substrate stream → Recursion projection** — `s_w = s_proj(mean_readout)` from the K substrate readouts.
3. **Recursion state before the tick** — `h` snapshot taken immediately before `recursion.step`.
4. **Recursion state after the tick** — `h` immediately after `recursion.step`.
5. **Recursion delta** — `‖h_after − h_before‖₂`.
6. **Single broadcast** — `h_cond` (the held state for the window).
7. **Both streams consume the broadcast** — `transformer_consumed_broadcast=True` (via `transformer.inject(hidden, inject_signal)`) and `substrate_consumed_broadcast=True` (via `x_i = emb_in[:, i, :] + cond_in` where `cond_in = cond_proj(h_cond)`).
8. **Certificate margin at this boundary** — `self.recursion.audit()["margin_h"]`.

All tensors are `.detach()`-ed before summarisation. Norms are computed
under `torch.no_grad()`. Raw text is never captured; `source_record_ids`
defaults to an empty tuple.

## Fixture identity for the L1 tests

- **Fixture**: bounded synthetic-English inputs (`torch.randint`) — permitted per the corpus staging rule for L0–L2 (implementation only, no scientific claim).
- **Tokenizer identity**: N/A at L1 — the observer sits below the tokenizer.
- **Corpus identity**: N/A at L1.
- **Model configuration**: h_rec=64, K=16, hidden_size=64, num_attention_heads=2, seq_len ∈ {24, 32, 48}.
- **Checkpoint identity**: L1 does not exercise a persisted checkpoint. Resume continuity across a persisted protected generation is provable by the W10-11 harness; L1's tests only prove the observer is byte-identical to observer=None.

## Number of traced windows

Depending on the test fixture, {2, 2, 3} boundaries — enough to prove
the numbering scheme, the K=16 alignment, and the short-final-window
handling. The tests are structural, not scientific.

## Noninterference result

**PASS.** Every noninterference test in
`tests/test_l1_signal_trace.py` runs:

- `test_observer_none_vs_null_observer_produces_identical_output` — logits and loss bit-identical between `observer=None` and `observer=_NullObserver()`.
- `test_observer_does_not_change_gradients` — post-backprop gradients agree exactly (`atol=0.0`, `rtol=0.0`) between the two forward paths.
- `test_observer_does_not_mutate_model_parameters` — model parameters immediately before and immediately after the observer-active forward compare equal.

The noninterference test uses `torch.random.set_rng_state(rng_before)`
between the two forward calls to keep the transformer's internal RNG
consumption identical across the pair.

## Instrumentation overhead

The observer path adds:

- One `if observer is not None:` branch check per window boundary.
- Six `.detach().float().norm()` computations per boundary (t_w, s_w, h_before, h_after, h_cond, and the delta between h_before and h_after).
- One `self.recursion.audit()` call per boundary.
- One dataclass construction per boundary.
- One `observer.on_recursion_window(event)` call per boundary.

At K=16 and typical seq_len (32–2048), the boundary count is small
(2–128 per forward), so overhead is bounded to microseconds per
boundary. No timing budget is exceeded on the test fixtures.

**Overhead when `observer=None`**: exactly one branch check per
window. Zero allocations, zero norm computations, zero dataclass
constructions.

## Architecture-preservation result

**PASS.**
`tests/test_ip_preservation.py` runs first in the regression and
enforces:

- K=16 declaration intact.
- Recursion still `.float()`-cast in the worker and the slow-clock tick.
- Exactly one `inject_cols.append` and one `transformer.inject` call in `forward`.
- `substrate.step` receives only `x_i` (no transformer entropy / logits / hidden states / attention state).
- `observer` / `intervention` kwargs default to None so probe-absent semantics stand.
- No outbound-network / third-party-upload references in `aeon/hybrid.py` or under `aeon/bypass/`.

## IP-preservation result

**PASS.** L1 lands only:

- A new module `aeon/bypass/signal_trace.py` (`RecursionWindowEvent`, `AeonDiagnosticObserver` protocol, `TensorCaptureBudget`, `_NullObserver` test helper, and small detached-norm helpers).
- Two additional optional keyword arguments on `HybridModel.forward`, both defaulting to None (`observer`, `intervention`). `intervention` is reserved for L5 and is currently ignored when supplied; L5 owns the training-guard check.
- A new test file.

No proprietary module or class is renamed, deleted, flattened, or
replaced. The default forward path is byte-for-byte identical. No
outbound-network or third-party-upload dependency introduced.

## Known limitations

- **Raw tensor capture at L1**: the `TensorCaptureBudget` dataclass is defined and defaults to disabled, but full raw-tensor capture is deferred to L4/L5 when the offline diagnostic runs need it. L1 delivers scalar summaries only.
- **Resume continuity trace**: L1's own tests do not spin up a full protected-generation resume cycle. The W10-11 end-to-end harness already proves the cycle without an observer; the observer noninterference tests prove the observer does not corrupt any resumed state. A dedicated cross-resume trace evidence run lands with L4 telemetry.
- **`source_record_ids` population**: at L1 the field always defaults to an empty tuple. L4 will populate it from the corpus batch source when telemetry is emitted from the worker path.
- **Claim level**: L1 remains at claim level 0 (`THEORY_ONLY`). L1 is strictly `STRUCTURALLY_IMPLEMENTED` — the ladder only advances after real-corpus observational evidence lands (Level 2+, gated on the vendored corpus package).

## Claim ladder position

L1 does not claim a bypass, does not attempt causal inference, does
not run interventions, does not report barrier-relative statistics.
It only proves the machinery is wired.

Achieved claim level after L1: **0** (unchanged).
Achievable at L1 completion: 1 (`STRUCTURALLY_IMPLEMENTED`).
Actual value in `docs/latent_bypass/status.json.achieved_claim_level`:
kept at 0 until L2 also lands (level 1 is reported jointly by L1+L2).
