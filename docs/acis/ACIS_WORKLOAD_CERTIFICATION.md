# ACIS Workload Certification — under the P2 real-English checkpoint

**Checkpoint:** `runs/aeon_lbc1_P2/final.pt` — 512 steps, 1,048,576 useful
                 training tokens, 7.015 M params, sha256 recorded in
                 `docs/training/p2_evidence.json`.
**Trial workload:** 32 evaluation batches × 8 batch × 256 seq =
                    65,536 tokens per trial. 5 trials per ACIS mode.
**Environment:** CPU-only (4 threads), fp32.
**Runner:** `scripts/acis_workload_certify.py` (deterministic seeds).

Evidence: `docs/acis/acis_workload_evidence.json`.

---

## 1. Semantic equivalence (§13)

| Mode | Publishes / trial | Audit events / trial | Stacked-logit digest matches OFF | Gradient digest matches OFF |
| ---- | ----------------: | -------------------: | -------------------------------- | --------------------------- |
| OFF  | 0                 | 0                    | (baseline)                       | (baseline)                  |
| OBSERVE | 512            | 2,048                | **True**                         | **True**                    |
| BUCKET  | 512            | 2,048                | **True**                         | **True**                    |

OBSERVE and BUCKET publish one broadcast per K-boundary (32 batches ×
16 boundaries per batch = 512 broadcasts) and record four audit events
per boundary (publish + lease_issue + lease_ack + retire = 2,048 audit
events).

Both modes' stacked logit digests across all 32 batches equal the OFF
digest. Both modes' gradient digests on the fixed backward pass equal
the OFF digest. **No semantic drift.**

---

## 2. Wall-clock overhead

| Mode    | median (ms) | p95 (ms) | Overhead vs OFF (median) | Overhead vs OFF (p95) |
| ------- | -----------:| --------:| ------------------------:| ---------------------:|
| OFF     | 27,955      | 29,370   | (baseline)               | (baseline)            |
| OBSERVE | 30,023      | 30,609   | **7.40 %**               | **4.22 %**            |
| BUCKET  | 29,536      | 30,477   | **5.66 %**               | **3.77 %**            |

Peak transient memory (via `tracemalloc`):
* OFF: ~131.1 MB
* OBSERVE: ~133.3 MB (+2.2 MB per trial)
* BUCKET:  ~133.3 MB (+2.2 MB per trial)

The +2.2 MB is bounded by the shuttle's per-trial audit-log tail —
proportional to trial length, not per-forward. The `AcisEvent`
frozen dataclass carries no payload references (ACIS-0 invariant).

---

## 3. Certification decision (§13 gates)

### OBSERVE
* Target: median overhead < 3 %.  Measured: **7.40 %**  → **fails target**
* Hard ceiling: p95 overhead ≤ 5 %.  Measured p95: **4.22 %**  → passes ceiling
* Semantic equivalence: **passes**

**OBSERVE certification result: NOT CERTIFIED at the target — the
p95 ceiling passes but the median overhead exceeds 3%.** The gap is
small in absolute terms (~2 s over a 28 s workload) but the certified
gate is unmet on this workload.

### BUCKET
* Target: median overhead < 3 %.  Measured: **5.66 %**  → fails target
* Hard ceiling: p95 overhead ≤ 5 %.  Measured p95: **3.77 %**  → passes ceiling
* Semantic equivalence (logits, loss, gradients): **passes**
* Default-eligibility (real benefit under previously certified policy):
  requires median overhead ≤ 0. Measured: 5.66 %. **Not default-recommended.**

### CONVEYOR_EXPERIMENTAL
* Not measured under this workload.
* Certified default from ACIS-8: `conveyor_refused / no_conveyor_evidence`.
* **REFUSED** — unchanged.

---

## 4. Interpretation

The overheads are real signal, not measurement noise: OBSERVE and
BUCKET both add ~5–7 % wall time on this specific workload
(small model, CPU, fp32, 65k tokens per trial, 512 boundaries per
trial). The shuttle's per-boundary work is:

  * one `publish_broadcast` (semantic digest = SHA-256 over the
    contiguous fp32 view of h_cond — 128-element vector × 8 batch = 4 KiB / boundary)
  * one lease pair issue
  * one audit-log append (chained SHA-256 over metadata only)
  * one lease pair acknowledge
  * one retire

None of that is asymptotically expensive; it is small O(1)-per-boundary
Python overhead + one SHA-256 per boundary. On a small CPU model where
the forward itself is only ~55 ms per batch, that Python overhead is
proportionally visible.

The ACIS-0 §12 baseline recorded 0 clones and 0 device transfers in
default OFF — that invariant still holds; the overhead is book-keeping,
not tensor duplication.

**No modification to K, model semantics, loss, or the substrate gate
was proposed or performed.**

---

## 5. Recommendation

* **Keep OFF as the certified default.**
* **Do not enable BUCKET by default.** Semantic equivalence is proven
  but the observed 5.66 %-median-overhead does not meet the certified
  benefit gate under this workload.
* Re-measure OBSERVE and BUCKET on the full-scale target hardware
  (GPU, larger model, higher token budget per trial) before revisiting
  BUCKET-default eligibility. The certified gates and the runner are
  now in place to make that re-measurement one command.

---

## 6. Evidence

Full trial-level records live in `docs/acis/acis_workload_evidence.json`.
Every wall-time is a real `time.time()` measurement; every digest is a
real SHA-256 over the produced tensor bytes.

**Nothing about the ACIS-8 certification is being renegotiated by this
report.** ACIS-8 landed under `bucket_certified: false` and
`conveyor_certified: false`; those flags remain false.
