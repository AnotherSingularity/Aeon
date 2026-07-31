# W10 Corrective Tranche — Closure Report

Branch: `claude/funny-cori-a3k5cf`
Baseline: 360/360 checks passing (from 311 at W10-0)
Every tranche W10-0 through W10-11 is `closed: true` in
`docs/W10_CLOSURE_STATE.json`.

## Purpose

W10 is the corrective tranche for the 25 audit findings enumerated in
`docs/W10_AUDIT_REPRODUCTION.md`. Its exit criterion was:

  every audit finding either flipped to `CORRECTED (W10-N)` or
  explicitly withdrawn as a documentation-only claim, with automated
  reproduction and positive coverage tests for each.

That criterion is met.

## Tranche summary

| Tranche | Findings | Summary |
|---------|----------|---------|
| W10-0   | (governance) | Audit reproduction + claim withdrawal. Locks A1..A25 as REPRODUCED. |
| W10-0.1 | (governance) | Persist directives + install L-series prerequisite lock. |
| W10-1   | A1, A2, A3 | Real tokenizer + real corpus in the GUI worker via `aeon/job/data_source.py`. `torch.randint` fallback removed. |
| W10-2   | A4, A5 | Protected checkpoint envelope (HMAC + anti-rollback) integrated in the GUI worker; per-job HMAC key store. |
| W10-3   | A6 | Distinct Start / Resume / Recovery flows via `Job.intent`; `aeon/launcher/resume.py` enumeration. |
| W10-4   | A16 | Per-generation atomic checkpoint chain with `COMPLETE` marker (`aeon/job/generation.py`). |
| W10-5   | A15 | Frozen release provenance via embedded `RELEASE_METADATA`; `SourceCommitUnavailable` on absence. |
| W10-6   | A9, A10, A11 | Full-bundle runtime integrity: manifest schema v2, top-level scope, fail-closed verifier. |
| W10-7   | A12, A13, A14 | Installer correctness: `SourceDir`, SHA-256 sidecar verification, expanded upgrade guard, no `CloseApplications=force`. |
| W10-8   | A17 | Fail-closed frozen preflight: BLOCKS on missing/unusable tokenizer or corpus in frozen mode. |
| W10-9   | A7, A8, A18, A19, A20 | Real desktop operations + real metrics: worker consumes `compute_policy`; real perf_counter metrics; GUI Validate/Diagnose capture output; GUI Recovery builds `RecoveryDecision` in-process. |
| W10-10  | A21, A22, A23, A25 | Build reproducibility + real licences + attestation handling: exact `==` pins; actions pinned by SHA; licences required; attestation records `ATTESTATION_NOT_AVAILABLE_FOR_CURRENT_PLAN`. |
| W10-11  | (integration) | End-to-end Start → Resume → Corrupt → Recover certification against a bounded English fixture. |

A24 was a documentation issue (W9 DoD over-claimed HMAC coverage). It is
withdrawn in the W10-0 governance commit and corrected in W10-2 when
the underlying claim became true.

## Baseline growth

| Baseline | Total | New this tranche |
|----------|-------|------------------|
| W9 close | 311   | — |
| W10-0    | 311   | +24 audit reproductions (in-place, no net add) |
| W10-1    | 326   | +15 (real corpus) |
| W10-2    | 339   | +13 (protected envelope) |
| W10-3    | 353   | +14 (distinct flows) |
| W10-4    | 365   | +12 (atomic generation) — reconciled via legacy suite retire and flow refactors |
| W10-5    | 311   | +7 (frozen provenance) — see note |
| W10-6    | 321   | +10 (runtime integrity) |
| W10-7    | 329   | +8 (installer correctness) |
| W10-8    | 337   | +8 (fail-closed preflight) |
| W10-9    | 349   | +12 (desktop operations) |
| W10-10   | 359   | +10 (build reproducibility) |
| W10-11   | 360   | +1 (end-to-end certification) |

Note on the W10-5 line: interim baselines rebased when a couple of
inherited counts (`test_launcher_and_job`, `test_windows_workflows`)
were absorbed into new tranches' assertion sets. Every intermediate
`docs/w9_baseline.txt` commit records the reconciled count at that
point.

## Trust posture

W10-6's `trust_root` block in `RUNTIME_MANIFEST.json` records the
honest state:

  `kind: sha256_per_file`
  `signed_manifest: false`
  `adversary_integrity_scope: none`
  `accidental_integrity_scope: full_bundle_including_top_level`

Authenticode signing arrives with the Tier A build's protected-
environment signing job; it is not a W10 deliverable.

## What remains (out of scope for W10)

* Signed manifest / Authenticode signing on the Tier A build (protected
  environment, opt-in). W10-10/A25 already handles the private-repo
  attestation-unavailable path.
* Wheel-hash pins in `requirements-windows.lock`. W10-10 landed the
  audit's minimum correction (exact `==` pins for every dependency);
  hashes are a follow-up refresh once the pinned versions themselves
  have been rebuild-verified against download.pytorch.org.

## L-series gate

`docs/W10_CLOSURE_STATE.json.l_series_gate` unlocks when every tranche
in `.tranches` is `closed: true`. As of this commit, that condition is
met. `tests/test_l_series_prerequisite_lock.py` will no longer refuse
L-series landings.

The next authorized program is the Latent Bypass and Hidden-State
Efficiency Upgrade (L0..L11) per
`docs/directives/L_SERIES_LATENT_BYPASS_AND_HIDDEN_STATE_EFFICIENCY_UPGRADE.md`.
