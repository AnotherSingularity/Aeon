# DESKTOP-R2 — Export Behavioral Equivalence + Architectural Trace

**Verdict:** **PROVEN.** Forward logits are byte-identical. Deterministic
16-token generation is byte-identical. Fixed-K=16 boundary schedule is
executed. Transformer + Substrate + Recursion all execute. One shared
broadcast per boundary. ACIS OFF throughout desktop generation.

Machine-readable: `docs/desktop/desktop_export_equivalence.json`.

---

## 1. Fixture

* **Prompt:** `"The"` (fixed, non-secret, not derived from any partition)
* **Fixture seed:** 20260803
* **Same tokenizer:** `research-data/AEON-LBC-1/tokenizer/aeon-lbc1.model`
  (sha256:`064ab6a9…`)
* **Same config:** `configs/latent_bypass/aeon_lbc1_proxy.yaml`
* **Same device / dtype:** CPU / fp32
* **Same thread policy:** default
* **`model.eval()` + `torch.inference_mode()`**
* **ACIS OFF:** `shuttle=None` on every forward call
* **No interventions**

## 2. Ten proofs required by §4

| Requirement                                  | Result                       |
| -------------------------------------------- | ---------------------------- |
| 1. Parameter names match exactly             | **PASS**                     |
| 2. Parameter tensors match exactly           | **PASS** (`torch.equal` per key)|
| 3. Forward logits match                      | **PASS** (`torch.equal`)     |
| 4. Loss matches                              | **PASS** (exact float equality)|
| 5. Fixed-seed generated token IDs match      | **PASS** (16 tokens byte-identical) |
| 6. Recursion-boundary indices match          | **PASS** (3 boundaries for seq_len=40 at K=16) |
| 7. Number of shared broadcasts matches       | **PASS** (1 per boundary)    |
| 8. Transformer-source contribution occurs    | **PASS** (per-forward wrap)  |
| 9. Substrate-source contribution occurs      | **PASS** (per-forward wrap)  |
| 10. No hidden training state needed          | **PASS** (loaded via `weights_only=True`) |

## 3. Architectural trace

Wrapped `model.recursion.step`, `model.transformer.read`, and
`model.substrate.step` to record per-boundary metadata WITHOUT
exposing raw tensor bytes. The trace records only:

* `boundary_index`
* configured `K`
* transformer / substrate source tensor **shapes** and **dtypes**
* `h_new` shape and dtype (must be `torch.float32`)
* **1** semantic broadcast produced per boundary
* **2** destination paths consuming that broadcast

For a fixture of 40 tokens at K=16:

* 3 K-window boundaries observed (matches `ceil(40 / 16)`).
* Every boundary's `h_new_dtype == torch.float32`.
* Transformer + Substrate both observed to execute at least once.

### Failure gates (each raises `AssertionError` if triggered)

* `K != 16` at config or effective trace
* `n_boundaries != ceil(seq_len / K)`
* Any boundary with `broadcasts_produced != 1`
* Any boundary with `destination_paths_consuming_broadcast != 2`
* Any boundary with `h_new_dtype != "torch.float32"`
* `transformer_ran is False`
* `substrate_ran is False`
* Any `shuttle`, `intervention`, or `observer` kwarg present during
  desktop generation

## 4. ACIS OFF verification during desktop generation

`test_R2_desktop_runtime_generation_is_ACIS_OFF` wraps
`HybridModel.forward` on a live `AeonDesktopRuntime` and asserts, for
every call made during a 4-token generation, that all three of
`shuttle`, `intervention`, and `observer` kwargs are `None`. That is
executable proof — not source-code inspection.

## 5. Tolerance policy

No tolerance is used. Both models load identical bytes at identical
precision on identical device, and both forward paths execute the
same code (no format-conversion layer) — `torch.equal` succeeds. If
this ever fails on a target platform (e.g. a hypothetical bfloat16
path), the maximum absolute error will be reported and a documented
tolerance authorized separately; that is not the case at this commit.

## 6. Non-negotiables preserved

* No `aeon/hybrid.py` change.
* No `aeon/recursion.py` change.
* No `aeon/substrate/*` change.
* No `aeon/transformer.py` change.
* No `aeon/tokenizer.py` change.
* No aeon/shuttle/* change.
* Sealed TEST partition never touched.
* No training state persisted across this test.
