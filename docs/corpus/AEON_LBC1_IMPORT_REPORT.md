# AEON-LBC-1 Import + CORPUS-1 Preprocessing Report

**Corpus id:** `AEON-LBC-1`
**Base branch:** `claude/funny-cori-a3k5cf`
**Starting commit for this tranche:** `9f5433b` (validated intake package)
**Preprocessing version:** `aeon-lbc1-v1`
**Acquisition method:** `manual_official_download`
**Executing environment did not download:** `true`

---

## 1. Acquisition

Every source was supplied by the repository owner via a data-only upload
branch (`data/aeon-lbc1-offline-upload`), merged into the active branch
at commit `891819b`, relocated to the required intake paths at commit
`028fef6`, and validated + promoted by `scripts/vendor_aeon_lbc1.py`
at commit `9f5433b`. The importer performed no network access.

### Source digests

| eBook   | Work                                            | Partition role | Source bytes | SHA-256 (source bytes)                                                |
| ------- | ----------------------------------------------- | -------------- | ------------ | --------------------------------------------------------------------- |
| PG-11   | Alice's Adventures in Wonderland                | train          | 174,311      | `01b38ea4c710a84bc18d0bd41271a5a1a92b94e97b2812f4dece97d4a694725e`     |
| PG-55   | The Wonderful Wizard of Oz                      | validation     | 237,232      | `969bffab7740d4d8a0bac332d78bd0152ad4a40a2efc52f06c74a9bb6120be75`     |
| PG-84   | Frankenstein; Or, The Modern Prometheus         | calibration    | 448,885      | `7810cd483cffcf2cc8a1d8f0d5807931e69d4f48cd14149b8c76f88af82fead3`     |
| PG-1342 | Pride and Prejudice                             | train          | 772,386      | `74f2665d6e6925fc2c17dec644bec9e87df478a0f1836822125e8acbb3777806`     |
| PG-1661 | The Adventures of Sherlock Holmes               | **test** (sealed) | 607,606      | `922e2a12ccb43a4c9544c260b2166c6ad2097aeb5957faeee113f173bb857cd0`     |
| PG-2701 | Moby-Dick; Or, The Whale                        | train          | 1,276,263    | `9a6844ac0703853720010787c7b6c70b0020f1ab1862dcd74452fa46474d1215`     |

---

## 2. Preprocessing (§5)

`scripts/prepare_aeon_lbc1.py` applied the frozen `aeon-lbc1-v1`
transformations to every source:

1. Strict UTF-8 decode
2. BOM strip (only if present)
3. CRLF/CR → LF normalization
4. Unicode NFC
5. Detect Gutenberg header + footer markers; body extracted between them
6. Chapter indexing (stable identifier per detected chapter marker)
7. Paragraph split on runs of ≥ 2 LFs
8. Deterministic `record_id = SHA-256(work_id | chapter_id | paragraph_index | text)`

No source was normalized, modernized, or model-cleaned. Spelling,
punctuation, capitalization, and paragraph order are preserved.

### Preprocessed record counts

| Work / partition          | Records | Applied transforms                                                          |
| ------------------------- | ------- | --------------------------------------------------------------------------- |
| pg-2701 → train           | 2,802   | normalize_line_endings, boundary_stripped, chapter_indexed, paragraphs_split |
| pg-1342 → train           | 2,509   | normalize_line_endings, boundary_stripped, chapter_indexed, paragraphs_split |
| pg-11 → train             |   828   | normalize_line_endings, boundary_stripped, chapter_indexed, paragraphs_split |
| pg-84 → calibration       |   821   | normalize_line_endings, boundary_stripped, chapter_indexed, paragraphs_split |
| pg-55 → validation        | 1,164   | normalize_line_endings, boundary_stripped, chapter_indexed, paragraphs_split |
| pg-1661 → **test (sealed)** | 2,546 | normalize_line_endings, boundary_stripped, chapter_indexed, paragraphs_split |

### Partition digests (SHA-256 of the emitted JSONL bytes)

| Partition   | Records | Bytes     | SHA-256                                                             |
| ----------- | ------- | --------- | ------------------------------------------------------------------- |
| train       |  6,139  | 4,582,224 | `5c33cbcbe0e4ca6ad84bd6d27a751f1791e504945f82eee108f27ba4d7b07c59` |
| calibration |    821  |   769,050 | `bf7f722d9c8778ccf9ca445457b3c5c3bc2c4fcd78994ee4a39b39cdb110c1eb` |
| validation  |  1,164  |   683,753 | `ed75c7a4408d7839b2400b8cae2aaee17769aea28e57965edce3b26b60255b35` |
| **test** (sealed) | 2,546 | 1,620,097 | `a64f4cb9673f8b867cc81b041bda8198a6c403921ccf27f5546c32930e35947e` |

The four processed JSONL files live under
`research-data/AEON-LBC-1/processed/` on the local machine. That
directory is `.gitignore`-excluded because the sealed test partition
(PG-1661) must not travel with the repository — its identity + digest
appears in `docs/corpus/aeon_lbc1_manifest.json`, its text does not.
Every processed partition is reproducible from the committed source
bytes plus `scripts/prepare_aeon_lbc1.py` at fixed
`preprocessing_version = aeon-lbc1-v1`.

---

## 3. Sealed-test discipline (per addendum §2)

PG-1661's processed record file exists on disk but is not committed
and will not be:

* not printed, sampled, summarized, tokenized-for-fitting, scored, or
  debugged-against;
* only its identity, byte count, record count, source digest, partition
  digest, and schema validity appear in evidence artifacts.

Test access is gated by a committed
`docs/latent_bypass/L3_CALIBRATION_LOCK.json` per §17 of the parent
directive; that lock is not yet in place. Until it lands, the test
partition is read only by the coming CORPUS-2 leakage check (source
identity + digest math, no text exposure) and by L5.

---

## 4. Regression accounting (per addendum §1)

* **Command used for the 627-check figure:**
  a Python subprocess loop over `sorted(glob.glob('tests/test_*.py'))`,
  running each file with `python <file>` and parsing the `"N checks passed"`
  line each file's `_run_all()` writes. `pytest` is not installed in this
  container; each test file is self-contained and drives its own runner.
* **Collected totals:** 55 test files, 627 explicit checks, 0 failures.
* **Attribution:** the 616 → 627 delta arrived at commit `4c3c16e`
  (ACIS-8 closure, `tests/test_acis_8_closure.py` = 11 checks), which is
  an ancestor of `7b0c2f9`. No test files were added or modified by the
  three corpus-integration commits (`891819b`, `028fef6`, `9f5433b`);
  `git log --oneline 7b0c2f9..HEAD -- 'tests/**'` is empty.
* **Baseline classification:** **627 is the authoritative current
  baseline.** The earlier "616" report in the CORPUS-0 State-A summary
  was a stale carry-over from the pre-ACIS-8 total.

---

## 5. Aeon architecture + IP preservation

None of CORPUS-1 touched `aeon/`, `configs/`, tests, ACIS invariants,
K=16, the six V0.02.02 patches, or the shuttle default mode. Only the
data package + preprocessing manifest + this report + machine-readable
corpus manifest + the `.gitignore` exclusion for the derivable
`processed/` directory changed.
