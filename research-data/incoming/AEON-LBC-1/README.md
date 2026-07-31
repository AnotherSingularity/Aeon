# AEON-LBC-1 offline intake

Program B is currently **AWAITING_OFFLINE_CORPUS_SOURCES**. This
directory is the six-file boundary at which the repository stops until
you drop in the required inputs. Once every file described below is
present, `scripts/vendor_aeon_lbc1.py --source-dir <THIS_DIR>` will
validate them, atomically promote them into `research-data/AEON-LBC-1/`,
and Program B resumes automatically through corpus construction, P0–P2
training, and L3–L5.

## Required files

Place six official Project Gutenberg UTF-8 plain-text files under
`sources/`:

| eBook | Work | Filename |
|------:|------|----------|
| PG-11   | Alice's Adventures in Wonderland | `sources/pg-0011.txt` |
| PG-55   | The Wonderful Wizard of Oz | `sources/pg-0055.txt` |
| PG-84   | Frankenstein; Or, The Modern Prometheus | `sources/pg-0084.txt` |
| PG-1342 | Pride and Prejudice | `sources/pg-1342.txt` |
| PG-1661 | The Adventures of Sherlock Holmes | `sources/pg-1661.txt` |
| PG-2701 | Moby-Dick; Or, The Whale | `sources/pg-2701.txt` |

Do NOT use mirrors, scraped copies, HTML-to-text conversions,
audiobook transcripts, or unofficial modernized editions. The
importer validates each file's own contents (Gutenberg markers,
title evidence, strict UTF-8) — filename alone is not proof.

## Required provenance sidecars

For every source file above, place a companion JSON sidecar under
`provenance/`, named identically but with `.json` extension. Use
`provenance/PROVENANCE_TEMPLATE.json` as the starting point.

## Command that resumes execution

```bash
python scripts/vendor_aeon_lbc1.py \
  --source-dir research-data/incoming/AEON-LBC-1 \
  --package-root research-data/AEON-LBC-1
```

If any file fails validation, NO package state changes — the current
`research-data/AEON-LBC-1/` (if any) remains authoritative. Rerun
after the failing input is corrected.

## What the importer will NOT do

- Attempt network access.
- Modify the incoming files.
- Overwrite the previous package on any validation failure.
- Loosen source-authentication rules to accommodate a specific file.
- Copy corpus text into `LICENSE`, `README.md`, or the Windows
  installer's `[Files]` block.

## Preserving Aeon IP

The manually supplied corpus is input data only. It does not alter
the ownership or license of Aeon source, architecture, weights,
checkpoints, algorithms, evidence, runtime, or installer. Corpus
license text remains in `research-data/AEON-LBC-1/license/` and is
never pasted over Aeon repository files.
