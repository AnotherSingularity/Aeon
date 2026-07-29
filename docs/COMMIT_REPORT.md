# V0.02.03 Architecture-Preserving Efficiency Upgrade — Commit Report

## Branch and commits

- **Repository:** `AnotherSingularity/AeonV0.02`
- **Original commit (V0.02.03 tip):** `d40ee13`
- **Working branch:** `claude/funny-cori-a3k5cf` (fast-forwarded to V0.02.03 at
  E0 start; every E0–E7 commit is additive on top of `d40ee13`, no history
  rewrite, no force-push, no rebase, no amend)
- **Final commit (E7):** _pushed at end of this phase_

### Ordered checkpoint commits

| checkpoint | commit | subject |
|---|---|---|
| E0 | `fd87b8d` | Repository & preservation audit |
| E1 | `d510f76` | Architectural contract tests |
| E2 | `568a395` | Low-overhead observability |
| E3 | `9b8d21c` | Checkpoint, resume, security hardening |
| E4 | `0bfde47` | Offline diagnostics |
| E5 | `b9aaf4e` | Runtime & overhead certification |
| E5' | `cc034dc` | Hygiene: gitignore runs/, remove artefacts from git |
| E6 | `1a1fd90` | Primary-training readiness |
| E7 | (this commit) | Final documentation & closure |

Every commit was authored with `Co-Authored-By: Claude Opus 4.7 …` and the
Claude-Session URL. No commit signs hooks were skipped, no `--no-verify` was
used, no destructive git operation was performed.

## Files changed vs the V0.02.03 base

**+3,737 / −31 lines across 25 files:**

### Added

- `aeon/checkpoint.py` — atomic save + strict load + integrity metadata (E3).
- `aeon/observability.py` — Observer + JSONL writer + static accounting (E2).
- `configs/aeon_350m_primary.yaml` — immutable primary campaign config (E6).
- `configs/aeon_smoke_e5.yaml` — bounded certification config (E5).
- `docs/E0_REPOSITORY_AUDIT.md`, `PRESERVATION_MANIFEST.md`, `TOPOLOGY_MAP.md`,
  `preservation.json`, `topology.json`, `baseline_tests.json` (E0).
- `docs/E5_CERTIFICATION.md`, `e5_evidence.json` (E5).
- `docs/OPERATIONS.md`, `e6_parameter_accounting.json` (E6).
- `docs/SECURITY_MODEL.md`, `PROXY_CAMPAIGN_PLAN.md`, `COMMIT_REPORT.md` (E7).
- `scripts/diagnose.py` — offline diagnostic tool (E4).
- `scripts/e5_certify.py` — bounded certification runner (E5).
- `tests/test_six_patches.py` (E1, 6 checks).
- `tests/test_recursion_topology.py` (E1, 6 checks).
- `tests/test_stream_independence.py` (E1, 5 checks).
- `tests/test_config_invariants.py` (E1, 5 checks).
- `tests/test_observability.py` (E2, 5 checks).
- `tests/test_checkpoint.py` (E3, 9 checks).
- `tests/test_diagnose.py` (E4, 2 checks).

### Modified

- `scripts/train.py` — observability integration, atomic-save / strict-load
  wiring, data_position tracking, RNG state save/restore. Precision anchors
  and six-patch code paths are UNCHANGED.
- `.gitignore` — ignores `runs/`, `tokenizer/`.

### Unchanged (verified)

- `aeon/hybrid.py` — no topology change.
- `aeon/transformer.py` — six-patch anchors intact.
- `aeon/recursion.py` — certificate + fp32 policy intact.
- `aeon/substrate/*.py` — reset dtype policy + feedback autonomy intact.
- `aeon/diagnostics.py`, `aeon/tokenizer.py`, `aeon/data.py` — unchanged.
- `configs/aeon_350m.yaml`, `configs/aeon_v1.yaml` — unchanged.

## Tests added / modified

7 new test files, 38 new checks (per-file totals):

| file | checks |
|---|---:|
| `tests/test_six_patches.py` | 6 |
| `tests/test_recursion_topology.py` | 6 |
| `tests/test_stream_independence.py` | 5 |
| `tests/test_config_invariants.py` | 5 |
| `tests/test_observability.py` | 5 |
| `tests/test_checkpoint.py` | 9 |
| `tests/test_diagnose.py` | 2 |
| **subtotal new** | **38** |

Pre-existing tests, all still passing (per-file totals):

| file | checks |
|---|---:|
| `tests/test_substrate_port.py` | 5 |
| `tests/test_aeon_sanity.py` | 6 |
| `tests/test_tokenizer.py` | 2 |
| `tests/test_feedback.py` | 5 |
| `tests/test_feedback_diagnostics.py` | 5 |
| **subtotal existing** | **23** |

## Full-suite totals

**Total: 61 / 61 pass, 0 fail.** No pre-existing failure was hidden; the E0
baseline recorded 23/23 pass at the outset, and the final state is 61/61 with
zero regressions.

## Known limitations

- **CPU-only certification environment.** The runtime figures in
  `docs/E5_CERTIFICATION.md` were measured on 4-thread CPU. The primary
  350.28M training run is expected on GPU hardware and will re-measure.
- **Synthetic-token pipeline path.** The certification runs use random-token
  synthetic data because the multilingual corpus is on Dylan's track. The
  primary launch will exercise the tokenized-corpus branch of
  `scripts/train.py`, which is covered by the same tests + observability but
  has not been end-to-end run at scale in this repository.
- **Overhead measurement noise on tiny models.** The E5 scenario reported
  overhead as negative (−27.6 %) because tiny CPU step-times sit inside OS
  scheduling noise. The dedicated E2
  `test_permanent_instrumentation_overhead_under_15_percent` asserts the 15 %
  ceiling with warm-up + median-of-N and passes at worst-case sample_every=1.
- **No live corpus present.** `data.tokenizer` and `data.corpus` in the
  primary config are placeholders. The launch procedure in
  `docs/OPERATIONS.md` explicitly requires operator fill-in — because the
  tokenizer identity is part of `strict_load`'s compatibility gate, a mistake
  cannot silently corrupt future resumes.

## Exact command for primary training

```bash
# 1) fill data.tokenizer + data.corpus in a working copy of configs/aeon_350m_primary.yaml
# 2) launch:
python scripts/train.py --config configs/aeon_350m_primary.yaml
```

Full launch and recovery procedures are in `docs/OPERATIONS.md`.
