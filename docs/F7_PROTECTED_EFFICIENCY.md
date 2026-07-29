# F7 — Protected Efficiency Certification

**Runner:** `scripts/f7_certify.py`. **Evidence bundle:** `docs/f7_evidence.json`.

Aeon is measured **with the mandatory protection envelope active** (§F7.1). Efficiency figures obtained by disabling protection are NOT valid defensive-efficiency evidence and are not reported here.

## 1. Profiles (§F7.1)

Six profiles, run over the same small representative model (`configs/aeon_smoke_e5.yaml`), same fixed data seed, same CPU-thread settings:

| # | profile | measured cost / behaviour |
|---|---|---|
| 1 | Certified Aeon, optional observability disabled | baseline step time |
| 2 | Certified Aeon, normal observability enabled | +observability cost |
| 3 | Certified Aeon + artifact authentication + provenance | +provenance/authentication cost |
| 4 | Certified Aeon + complete protection envelope | +MAC + audit + containment enforcement |
| 5 | Declared DEGRADED mode | reduced seq_len (§F5.4) |
| 6 | Recovery verification | strict/protected checkpoint load |

## 2. Measurements (§F7.2)

Reported in `docs/f7_evidence.json` for every profile: median step time, mean step time, tail step time, step count, per-category auxiliary cost, checkpoint save/load duration, peak resident memory. Static accounting (parameter bytes, optimiser bytes estimate, Recursion/substrate state bytes, checkpoint bytes estimate) is captured once per run.

## 3. Cost separation (§F7.3)

Latest run (small CPU model — 4 threads — 16 measured steps after 4-step warm-up):

| category | seconds per step |
|---|---:|
| base architecture + framework | ~60 ms |
| observability | ~0 ms (noise-dominated at this scale; E2 test asserts < 15 % at worst case) |
| artifact verification (sha256 sidecar + provenance strict_verify) | ~1 ms per step |
| cryptographic (HMAC over payload + AEAD wrap on saves) | < 1 ms per step |
| audit (per-step hash-chained event write) | ~0.1 ms per step |
| runtime containment enforcement | ~0.2 ms per step |
| recovery: full protected load | ~10 ms per event |
| **total protected step time** | ~60 ms (≈ base at this scale; costs are noise-dominated) |

Costs are reported SEPARATELY — never combined into a single unexplained percentage. **On this scale, cryptographic + audit + containment per-step costs sit in the sub-millisecond band**; they will be more visible at larger step sizes on GPU where the base step-time is longer.

## 4. Optimization rule (§F7.4)

If protected overhead becomes excessive at larger scale, the permitted optimisations are:

1. Profile the precise source.
2. Remove redundant work.
3. Cache immutable verified identities safely (source_commit sha, tokenizer sha).
4. Batch audit writes (write once per N steps behind a small buffer).
5. Avoid repeated serialisation (reuse canonical bytes across identity constructors).
6. Improve I/O (mmap corpus shards; async metrics writes).
7. Reuse validated manifests where policy permits.
8. Retest.

The following MUST NOT be weakened for the benchmark: authentication, integrity verification, contractive certification, anti-rollback, containment, audit evidence, recovery validation.

## 5. Claim boundary (§F7.5)

Per the directive, the results support the following claim only:

> Aeon demonstrates measured protected efficiency at its current implementation scale.

The results **do not** support: frontier superiority, universal nation-state resistance, military certification, energy efficiency (no energy measurement), full-scale performance claims.

## 6. Exit gate

- [x] Protected performance measured reproducibly (`scripts/f7_certify.py`; fixed seed; same config across profiles).
- [x] Every mandatory protection remains active during profiles 3–5 (attested in `f7_evidence.json::protection_envelope_active_during_measurement`).
- [x] Cost categories separated (§F7.3).
- [x] Degraded-mode capability measured (profile 5).
- [x] Security and observability overheads not conflated.
- [x] Suite: 137 inherited + 0 new F7 test files (F7 is a MEASUREMENT phase; the runner is a CLI). Full regression maintained at 137/137.
