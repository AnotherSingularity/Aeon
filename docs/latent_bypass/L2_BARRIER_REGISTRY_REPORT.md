# L2 — Visible Barrier Registry Report

Base: L1 (`537c972`).
Registry file: `benchmarks/latent_bypass/barriers.json` (schema v1).
Loader / calibrator / evaluator: `aeon/bypass/barriers.py`.
Same-visible-state candidate search: `aeon/bypass/candidates.py`.
Corpus-package validator (L3+ prep): `aeon/bypass/corpus_package.py`.
Enforcement tests: `tests/test_l2_barrier_registry.py` (16 checks).

## What L2 delivers

L2 defines visible computational barriers as external evaluation logic.
Barriers do NOT enter the model. They cannot be routed through, cannot
feed the substrate gate, cannot appear in the transformer forward
policy, cannot re-weight training loss, cannot drive adaptive compute.
`aeon/bypass/barriers.py::FORBIDDEN_REGISTRY_INPUTS` names every
hidden-state field that the registry loader will refuse to accept —
enforced at load time by a recursive walk over the JSON payload.

L2 also delivers the same-visible-state candidate-pair search that
L3+ uses to construct matched-visible-state comparisons **before** any
downstream code inspects the Recursion state. Locking the candidate
set before hidden-state comparison is what prevents coordinate
p-hacking (choosing candidates because they happen to produce a
favourable hidden-state result).

And L2 delivers the corpus-package validator machinery so an L3
calibration cannot begin until a real-English vendored corpus is
present, well-formed, and sealed.

## Definitions in `benchmarks/latent_bypass/barriers.json`

Eight rows land at L2:

| barrier_id | visible metric | threshold method | applicable tasks |
|---|---|---|---|
| HIGH_LOCAL_LOSS | pre_broadcast_token_loss | top_percent | all 8 |
| LOW_TARGET_MARGIN | pre_broadcast_correct_token_margin | bottom_percent | long_entity_recall, delayed_instruction, pronoun_resolution, nested_dependency |
| HIGH_VISIBLE_ENTROPY | pre_broadcast_output_entropy | top_percent | contradiction_resolution, state_tracking, local_aliasing, long_negation |
| LONG_DEPENDENCY | dependency_distance | per_partition_quantile | long_entity_recall, nested_dependency, long_negation |
| LOCAL_STATE_ALIASING | local_state_repetition | top_percent | local_aliasing, pronoun_resolution |
| CONTRADICTION_REGION | visible_failure_to_resolve_task_state | median | contradiction_resolution |
| DELAYED_INSTRUCTION_REGION | dependency_distance | per_partition_quantile | delayed_instruction |
| ENTITY_STATE_DISCONTINUITY | visible_prediction_instability | top_percent | state_tracking |

All `threshold_value` fields ship as `null` and are populated by
`BarrierRegistry.calibrate()` on the calibration partition. Once
calibrated, the threshold LOCKS: `already_calibrated` is raised on any
subsequent calibration unless the caller passes
`allow_recalibration=True` (visible-in-log override).

## Calibration procedure

1. Compute the barrier's visible metric across the calibration
   partition (never the evaluation / test partition).
2. Call `registry.calibrate(barrier_id, samples)`.
3. The registry checks `len(samples) >= row.minimum_samples`
   (200 by default per L2 rows).
4. The registry applies the row's `threshold_method`:
   - `top_percent` → top 5% cutoff.
   - `bottom_percent` → bottom 5% cutoff.
   - `median` → 50% cutoff.
   - `per_partition_quantile` → 90th percentile cutoff.
   - `fixed_value` → the value in the registry file, no calibration.
5. Returned `BarrierDefinition` carries the locked `threshold_value`.

## Synthetic fixture identity

L2 exercises the machinery on the bounded synthetic-English fixture
inherited from W10-11 and small hand-written token sequences. This
is permitted per the corpus-staging rule (L0–L2 implementation only).
**No observational, causal, efficiency, or bypass claim is derived
from L2.** `docs/latent_bypass/status.json.achieved_claim_level`
remains `0`. Test `test_l2_does_not_elevate_claim_level` enforces
this.

## Partition identities

Every registry row's `calibration_partition` differs from its
`evaluation_partition`. At L2 both are strings from the vendored
partition manifest (currently `"calibration"` / `"test"`); the actual
JSONL files are checked at L3 by
`aeon/bypass/corpus_package.py::validate_corpus_package`.

## Thresholds and candidate counts

L2 does not run barrier calibration on the real corpus (that is L3's
job) and does not lock a candidate set (that is L3's job). L2 proves:

- Calibration on synthetic samples produces the expected policy behaviour (unit tests).
- `find_exact_prefix_matches` yields a deterministic pair list over identical inputs (unit test).
- `find_projection_matches` respects the caller-supplied epsilon.
- `build_locked_set` produces a stable SHA-256 digest so downstream
  L3+ code can bind to a specific candidate set.

## Determinism result

**PASS.** Two consecutive `find_exact_prefix_matches` calls on the
same records produce identical pair lists (order and content). The
`build_locked_set` digest is deterministic given identical pairs.

## Noninterference result

**PASS.** L2 does not touch `HybridModel.forward`. No hidden-state
field appears in the registry schema, and the loader rejects any
JSON row that references a `FORBIDDEN_REGISTRY_INPUTS` string. The IP
preservation firewall (`tests/test_ip_preservation.py`) still passes.

## Claim limitation

L2 remains at claim level `0` (`THEORY_ONLY`). L2 completion joined
with L1 completion is achievable at claim level `1`
(`STRUCTURALLY_IMPLEMENTED`) once a vendored real-English corpus
exists to trigger the level-1 report. The claim level in `status.json`
does NOT advance without that corpus.

## Exact next gate

**L3** — hidden reaction coordinates. Requires:

1. A vendored real-English corpus package that
   `aeon/bypass/corpus_package.py::validate_corpus_package` accepts as
   `ready_for_L3=True`.
2. The candidate set locked before any hidden-state comparison.
3. Threshold locks for every barrier used in L3+ evaluation.

If the corpus package is not present at the L3 gate, execution stops
with a specific package-format error — L3 does NOT fabricate evidence
against the synthetic fixture.
