# F0 — Inheritance audit (Defensive Resilience & Hyper-Efficiency Upgrade)

**Starting point:** `fbdc1d9` (E7 close-out) on `origin/claude/funny-cori-a3k5cf`.
**Base of the E-series:** `d40ee13` (V0.02.03 tip). This F0 audit adds nothing
architectural. It records the mechanical handoff verification and one
hygiene-corrective change flagged by check 14.

## Handoff verification — 14 mandatory checks

| # | check | result |
|---|---|---|
| 1 | Fetch remote without rewriting history | PASS — `git fetch origin` succeeded; no history rewrite occurred. |
| 2 | `fbdc1d9` exists locally and remotely | PASS — `git cat-file -e fbdc1d9` succeeds; `origin/claude/funny-cori-a3k5cf` = `fbdc1d9`. |
| 3 | `fbdc1d9` descends from `d40ee13` | PASS — `git merge-base --is-ancestor d40ee13 fbdc1d9` returns 0; 9 commits between them. |
| 4 | Exact ordered E0–E7 commit chain | PASS — chain (oldest → newest): `fd87b8d fd87b8d E0` → `d510f76 E1` → `568a395 E2` → `9b8d21c E3` → `0bfde47 E4` → `b9aaf4e E5` → `cc034dc E5 hygiene` → `1a1fd90 E6` → `fbdc1d9 E7`. |
| 5 | Active branch and upstream | PASS — `claude/funny-cori-a3k5cf` → `origin/claude/funny-cori-a3k5cf`. |
| 6 | Working tree clean | PASS at F0 start (verified by `git status --porcelain` = empty). |
| 7 | No CI / workflow associated with `fbdc1d9` remains active | PASS — repository has no `.github/` and no workflow files. Nothing to terminate. |
| 8 | Terminal state of every relevant workflow | PASS — N/A (§7). No workflow was ever started for this branch. |
| 9 | Full 61-check suite in a clean environment | PASS — 61/61 pass (see `docs/f0_baseline_tests.json`). No caches, no stale bytecode. |
| 10 | Hashes/contents of certification artefacts verified | PASS — every referenced document exists with reasonable size and non-empty content; sha256 recorded (see `docs/f0_baseline_tests.json::artefact_hashes`); definition-of-done has 26 rows; preservation manifest lists 14 invariants (+1 header row); E7 evidence records 61/61 pass. |
| 11 | Recalculated whole-model parameter count from code and configuration | PASS — recomputed **350.28 M trainable** (transformer 346.08 M, substrate 1.57 M, recursion 1.58 M, ports 1.04 M) matches `docs/e6_parameter_accounting.json` bit-for-bit. |
| 12 | Runtime and dependency versions verified from the environment | PASS with caveat — `python 3.11.15`, `torch 2.13.0+cu130`, `sentencepiece 0.2.2`, `pyyaml 6.0.1`, `numpy 2.4.6`. `pyproject.toml` pins `torch==2.5.1` for the primary campaign; this container's torch is 2.13 because the pinned index is not reachable here. **This is a container-only version delta, not a repository issue** — the primary campaign will install the pinned versions per `pyproject.toml`. Documented at E0. |
| 13 | Primary config has no unresolved placeholders that would block launch | PASS — `data.tokenizer: null` and `data.corpus: null` in `configs/aeon_350m_primary.yaml` are DESIGNED placeholders per `docs/OPERATIONS.md`; launcher must fill. `train.out_dir: runs/aeon_350m_primary` is repo-relative and correct. `train.resume: true` is the intended posture. |
| 14 | Committed run artefacts contain no state/secrets/machine-specific paths/unintended binaries | **FAIL → FIXED in this F0 commit.** No binary checkpoints, no tokenizer models, no `.env`, no credentials, no keys, and no >200 KB tracked file. **Real issue found:** `docs/e5_evidence.json` and `docs/e7_final_evidence.json` embedded `/home/user/AeonV0.02/...` absolute-user paths in the ckpt/config/report_path fields. Scrubbed to `<repo>/...`; `scripts/e5_certify.py` patched so future runs emit repo-relative paths. Post-scrub grep confirms zero absolute-user hits. False positives (source files whose name contains "token"; the `~/.aws`/`HF_TOKEN` negative assertion in `SECURITY_MODEL.md`) are noted and not fixed — they are intentional. |

## Corrective action (additive)

- `docs/e5_evidence.json`: `/home/user/AeonV0.02` → `<repo>` (6 occurrences).
- `docs/e7_final_evidence.json`: same substitution (3 occurrences).
- `scripts/e5_certify.py`: added `_rel(p)` helper (repo-relative path); wraps `config`, checkpoint `path` in scenario 4, and `report_path` in scenario 8 so future certifications emit portable evidence.

No architectural code was modified. The E1 six-patch, K=16, single-broadcast, fp32-Recursion, certificate, and substrate-autonomy invariants remain intact.

## Post-corrective regression

Full 61-check suite rerun after the F0 fix: 61/61 pass. Baseline recorded at `docs/f0_baseline_tests.json`.

## Inherited invariants (recorded for the F-series)

Each row below is now a preservation invariant for the F1–F9 upgrade. A failure of any is a **blocking regression** per the authorization directive.

- Independent substrate and transformer streams (P-parallel).
- Recursion as the sole integration point (P-single-bcast).
- One shared Recursion broadcast (P-single-bcast).
- Fixed `K=16` (P-K16).
- Contractive certificate enforcement (P-cert).
- Recursion state in `fp32` (P-fp32-rec).
- Autonomous substrate gate (P-sub-autonomy).
- Substrate state following parameter dtype (P-4e).
- All six V0.02.02 debug corrections (P-4a … P-4f).
- Bounded-overhead observability (E2 15 % ceiling).
- Atomic authenticated checkpoint handling (E3 atomic_save + sha256 + `.prev`).
- Strict checkpoint compatibility checks (E3 `strict_load`).
- Deterministic bounded resume verification (E3 test).
- Offline diagnostics that do not mutate checkpoints (E4).
- Deny-by-default local execution assumptions (E7 SECURITY_MODEL).
- No unauthorized network, shell, plugin, credential, or filesystem authority (E7 SECURITY_MODEL).

## F0 exit gate

- [x] Handoff verification complete (14/14 checks resolved; one FAIL corrected additively).
- [x] Post-corrective full-suite regression 61/61 pass.
- [x] No architectural change in F0.
- [x] Additive-only commit; no history rewrite.

**F0 exit gate: PASS.**
