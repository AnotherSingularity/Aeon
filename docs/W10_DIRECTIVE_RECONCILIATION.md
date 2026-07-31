# W10-R — Directive Reconciliation

Base: `2a3f92d` (L0 close; Program B frozen pending this tranche).
Directive of record: the definitive W10-1 through W10-11 spec supplied
2026-07-31 plus the audit reproductions at `docs/W10_AUDIT_REPRODUCTION.md`.

Machine-readable matrix: `docs/w10_reconciliation.json`.
Enforcement test: `tests/test_w10_reconciliation.py`.

## Purpose

W10-R exists because the shipped W10-1..W10-11 tranches used different
module names, class layouts, and serialization formats than the
directive prescribes. That is not by itself a functional problem —
equivalent implementations are acceptable. But behavioural equivalence
was asserted, not proven; and a few requirements slipped through
without a direct test or a corresponding runtime check. W10-R is the
bounded corrective tranche that (a) maps every substantive requirement
to a real implementation and test, (b) classifies each row honestly,
and (c) corrects only real gaps.

## Classifications

- **BEHAVIORALLY_SATISFIED** — the requirement is met by a specific
  implementation and a specific test that exercises the required
  behaviour (not the source string).
- **EQUIVALENT_IMPLEMENTATION** — the requirement is met by a
  different implementation than the directive's example, but the
  functional outcome (behaviour, failure handling, state continuity,
  evidence) is the same. Cited when the difference is substantive
  enough to name; cosmetic-only differences use the next category.
- **EVIDENCE_GAP** — the behaviour is present in code but no test
  exercises the specific requirement; the fix is a new test only.
- **SUBSTANTIVE_FUNCTIONAL_GAP** — the behaviour differs materially
  from the directive; a code correction is required.
- **SUBSTANTIVE_SECURITY_GAP** — the difference weakens an integrity
  or authorization guarantee; requires a code correction.
- **COSMETIC_STRUCTURAL_DIFFERENCE** — different class name / file
  layout / JSON field names with no user-visible or security effect;
  no action required.
- **NOT_APPLICABLE_WITH_EVIDENCE** — the requirement does not apply in
  this environment (with justification).

A requirement is **not** marked satisfied merely because:

- A similarly-named test exists.
- The audit finding was corrected.
- The suite is green.
- Documentation claims completion.
- A source string or class exists.

The cited test must exercise the required behaviour.

## Gaps identified

Five rows flagged as substantive; three rows flagged as evidence-only.

### Substantive functional gaps

- **R6** — periodic validation runs on the in-flight training `out`
  without switching the model to eval mode; the "own partition" clause
  is partially addressed (own audit log, no optimizer step) but the
  eval-mode / no-grad clause is not. **Fix:** run a fresh forward pass
  under `torch.no_grad()` inside the periodic validation, with model
  swapped to eval mode and restored to train mode after.
- **R8** — worker silently rebinds `tcfg_model.vocab_size` to the
  tokenizer's vocab instead of failing closed on mismatch. **Fix:**
  raise `DataSourceError('tokenizer_vocab_mismatch')` before model
  construction. Preserve the current rebind only when the config's
  `vocab_size` is unset (0 or None).
- **R20** — Resume compares `expected_model_config` but does not
  compare release identity (`source_commit`). A checkpoint from
  release A resumed under release B loads silently. **Fix:** add
  optional `expected_release_identity` to `protected_load`; worker
  Resume passes it; worker Recovery does not (per the directive's
  authorized-recovery-under-older clause).
- **R26** — frozen preflight does not check that
  `RELEASE_METADATA['source_commit']` is present and non-`"unknown"`
  in frozen mode. A frozen bundle with stripped release metadata can
  reach `READY_WITH_WARNINGS`. **Fix:** add a `release_identity`
  preflight check; frozen mode makes it a blocker; source-tree mode
  reports `skip`.

### Evidence gaps

- **R17** — GUI button eligibility uses conditional-error UX rather
  than conditional-enable, which is functionally equivalent (no
  invalid operation reaches a worker spawn). No test documents the
  error paths. **Fix:** add a test asserting Resume / Recovery error
  paths return without spawning a worker when no candidate exists.
- **R31** — W10-11 positive matrix covers ~17/25 steps. **Fix:**
  extend the existing end-to-end test with explicit assertions on
  multiple K=16 boundary crossings, runtime-integrity verification of
  the generation tree, and architecture-preservation spot-check
  (K=16 declaration + fp32 recursion) at run boundaries. No new
  training runs added.
- **R32** — W10-11 negative matrix covers 1/25 cases directly; the
  other 24 are covered by scattered tests but not consolidated.
  **Fix:** add `tests/test_w10_11_negative_matrix.py` with one check
  per required negative case; each check either invokes the primitive
  or asserts the covering test exists. New primitives added only
  where required (vocab mismatch — covered by the R8 fix; cursor
  identity mismatch — covered by DataSourceError codes; incomplete
  generation refusal — covered by `list_generations(include_incomplete=False)`).

### Cosmetic-only differences (no action)

- Directive's `aeon/training_data.py` with `CorpusCursor` /
  `TokenBatchStream` vs shipped `aeon/job/data_source.py` with
  `TokenizedCorpusBatchSource`. Functional equivalence: fail-closed
  tokenizer/corpus resolution, position-integer resume, DataSourceError
  reason codes. **COSMETIC_STRUCTURAL_DIFFERENCE.**
- Directive's `aeon/checkpoint_store.py` with `ProtectedCheckpointStore`
  and `KeyProvider` protocol vs shipped `aeon/job/generation.py` +
  `aeon/job/key_store.py`. Functional equivalence: transactional
  generation dir, HMAC-authenticated envelope, per-job key file.
  **COSMETIC_STRUCTURAL_DIFFERENCE.**
- Directive's `JobMode` enum vs shipped `Job.intent: str` with
  runtime validation. **COSMETIC_STRUCTURAL_DIFFERENCE.**
- Directive's per-generation `payload.bin` / `metadata.json` /
  `authentication.json` / `provenance.json` / `digest.json` /
  `COMPLETE` split vs shipped `state.pt` + `state.pt.meta.json` +
  `COMPLETE`. All components live inside the atomic-rename target;
  transactional equivalence holds. **COSMETIC_STRUCTURAL_DIFFERENCE.**

## Actions taken during W10-R

Corrections land in additive commits on top of `2a3f92d`. Each action
appears in the machine-readable matrix's `action_ledger` with its
target file, change summary, and the test added. Summary:

| Action | Target | Test added |
|--------|--------|-----------|
| Correct periodic-validation semantics (R6) | `aeon/job/worker.py::_run_periodic_validation` | `test_periodic_validation_uses_eval_mode_and_no_grad` |
| Fail closed on vocab mismatch (R8) | `aeon/job/worker.py`, `aeon/job/data_source.py` | `test_worker_fails_closed_on_tokenizer_vocab_mismatch` |
| Release-identity compatibility on Resume (R20) | `aeon/protected_checkpoint.py`, `aeon/job/worker.py` | `test_release_identity_mismatch_rejected_on_resume` + `..._allowed_under_recovery` |
| Frozen preflight release-identity check (R26) | `aeon/config/preflight.py` | `test_frozen_preflight_blocks_on_missing_release_identity` |
| Extend W10-11 positive matrix (R31) | `tests/test_w10_11_end_to_end_certification.py` | in-place extension |
| Consolidate W10-11 negative matrix (R32) | `tests/test_w10_11_negative_matrix.py` (new) | ~25 checks |
| Add button-gating test evidence (R17) | `tests/test_w10_reconciliation.py` | `test_resume_error_path_does_not_spawn_worker` |

## Exit gate

W10-R closes when `docs/w10_reconciliation.json.closure_state` shows:

```json
{
  "reconciliation_complete": true,
  "substantive_open_gaps": 0,
  "evidence_open_gaps": 0,
  "program_b_authorized": true,
  "w10_reconciled_commit": "<commit>"
}
```

and all of:

- No `SUBSTANTIVE_FUNCTIONAL_GAP` / `SUBSTANTIVE_SECURITY_GAP` /
  `EVIDENCE_GAP` rows remain open.
- Every W10 requirement maps to an implementation and executable test.
- Full inherited regression passes (373 + W10-R additions).
- Architecture-preservation gates pass (K=16, fp32 Recursion,
  contractive certificate, six V0.02.02 patches).
- Working tree clean.
- Local and remote heads match.

After W10-R closes, Program B resumes at L1.
