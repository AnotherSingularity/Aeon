# E0 — Repository & Preservation Audit

**Phase:** E0 (Repository and preservation audit)
**Base commit:** `d40ee13` (V0.02.03 tip on `origin/V0.02.03`)
**Working branch:** `claude/funny-cori-a3k5cf`
**Scope:** audit only. No architectural changes in this phase.

## 1. Repository identity

| field | value |
|---|---|
| origin | `AnotherSingularity/AeonV0.02` |
| base branch | `V0.02.03` (unchanged, untouched, no rebase) |
| working branch | `claude/funny-cori-a3k5cf` (fast-forwarded to V0.02.03; additive E0–E7 commits will land here) |
| base commit | `d40ee13e571edb2c738d3609585e52b63bc89733` |
| working-tree state | clean at E0 start |
| tracked files | 31 (see topology map §2 for the map) |
| Python | 3.11.15 |
| torch | 2.13.0 (CPU install for this environment; the primary-training config pins torch==2.5.1 with cu124) |
| sentencepiece | 0.2.1 |
| pyyaml | 6.x |
| numpy | 2.4.6 |
| OS | Linux 6.18.5 x86_64 |
| CPU | 4 threads reported (this laptop-class env) |
| memory | 15 Gi total |
| current model config | `configs/aeon_350m.yaml` — 128k vocab, 24×1024, GQA kv=4, h_rec=512, K=16, adaptive feedback on, β=0.01 gate penalty. Trainable = **350.28M** (verified in prior turn against a live optimizer step). |
| tokenizer identity | Aeon-native SentencePiece, decision-locked at 128k multilingual with byte-fallback (see `aeon/tokenizer.py`). `.model` and `.vocab` are versioned alongside weights. |
| corpus identity | corpus is Dylan's track (multilingual, top-50 languages, CulturaX + Wikipedia + The Stack v2 + Gutenberg + arXiv + Aeon identity core). Not present in the repository. `data.tokenizer`/`data.corpus` are `null` in configs; training paths handle the null case as a synthetic-token smoke fallback. |
| checkpoint format (pre-E3) | plain `torch.save({"step", "model", "optim"})` in `scripts/train.py`. E3 replaces this with an atomic writer + integrity metadata + wider state. |

## 2. Topology map

See `TOPOLOGY_MAP.md`. Confirms Aeon's parallel-stream topology matches the directive's preservation rules:

- Substrate (`aeon/substrate/*`) and Transformer (`aeon/transformer.py::AeonTransformer`) are **independent forward paths** — no direct cross-stream reads.
- Both streams submit their authorized outputs to Recursion (`aeon/recursion.py::RecursionJoiner`) at the slow-clock boundary in `aeon/hybrid.py::HybridModel.forward`.
- **K=16** is fixed in `aeon/hybrid.py::HybridModel.__init__` and asserted in both YAML configs.
- Recursion emits **one broadcast** (`h_{w-1}` held across the K-token window). Both streams consume this **same broadcast** — substrate via `cond_proj`, transformer via `inject()`. No dual-head projections.
- Recursion runs **fp32** (`scripts/train.py:116 model.recursion.float()`, mirrored in `scripts/infer.py:43`).
- The **contractive certificate** (σ<margin_h/margin_c) is structural (`RecursionJoiner._build` clamps σ<MARGIN by construction via `sigmoid(s)·MARGIN·Cayley(A)·diag(tanh(d))`) and reported by `audit()`.
- The substrate feedback controller (V0.02.03 extension) reads **only substrate-internal signals** (its own readout rate of change) — no transformer state — so §3.6 substrate autonomy holds.

## 3. Preservation manifest

See `PRESERVATION_MANIFEST.md`. Ten invariants pinned to code locations and existing/planned regression tests. **All six V0.02.02 patches identified from code, not inferred:**

| # | patch | anchor | scope |
|---|---|---|---|
| 4a | γ fp32 recast after `model.to(dtype)` | `scripts/train.py:119`, `scripts/infer.py` | dtype transition |
| 4b | γ Parameter created `dtype=torch.float32` | `aeon/transformer.py:246` | initialization |
| 4c | `inject()` fp32 residual add | `aeon/transformer.py:266` | forward |
| 4d | `write_proj` random init (`normal_(std=0.02)`) not zeros | `aeon/transformer.py:241` | initialization |
| 4e | substrate cell `reset()` follows param dtype | `aeon/substrate/matrix_cell.py:98`, `vector_cell.py:53` | state reset |
| 4f | rotary `inv_freq` fresh fp32 per forward, no `register_buffer` | `aeon/transformer.py:94` | forward; `register_buffer` count in `transformer.py` = **0** (verified) |

## 4. Baseline test execution

Run at E0 start, all suites, torch 2.13 CPU:

| suite | checks | pass | fail | skip | notes |
|---|---:|---:|---:|---:|---|
| `tests/test_substrate_port.py` | 5 | 5 | 0 | 3 mock-only skips (torch-free path exercised) | conformance passes |
| `tests/test_aeon_sanity.py` | 6 | 6 | 0 | 0 | includes γ bf16-trap regression + no-external-library gate |
| `tests/test_tokenizer.py` | 2 | 2 | 0 | 0 | SentencePiece train + round-trip + special-id layout |
| `tests/test_feedback.py` | 5 | 5 | 0 | 0 | load bound, gate range/grad, gate-off exact reduction, gate-on bounded, certificate all modes |
| `tests/test_feedback_diagnostics.py` | 5 | 5 | 0 | 0 | each diagnostic catches its own failure mode |
| **total** | **23** | **23** | **0** | **3 (structural)** | **wall-clock ≈ 8.6s on 4-thread CPU** |

No pre-existing failures. No warnings that indicate protected-behavior drift.

## 5. E0 exit gate

- [x] Repository identity documented (§1)
- [x] Actual topology mapped (§2 → `TOPOLOGY_MAP.md`)
- [x] All six V0.02.02 patches identified at code locations, not inferred from the research zip (§3 → `PRESERVATION_MANIFEST.md`)
- [x] Baseline tests recorded (§4)
- [x] No preservation rule ambiguous (each invariant in the manifest has a code location + test target)

**E0 passes.** Proceeding to E1 (architectural contract tests).
