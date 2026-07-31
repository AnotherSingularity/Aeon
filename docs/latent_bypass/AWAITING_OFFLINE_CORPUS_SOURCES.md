# AWAITING_OFFLINE_CORPUS_SOURCES

Program B is halted at the six-file intake boundary. This is the
authorised stopping condition for a **STATE B — Genuine External
Blocker** at the corpus-acquisition step, now escalated to a
resumable "waiting" state because the offline-import path lands in
this commit. Program B resumes automatically the moment the six
official Project Gutenberg files appear in the declared intake
directory.

## Expected intake directory

```
research-data/incoming/AEON-LBC-1/
    sources/
        pg-0011.txt    (missing)
        pg-0055.txt    (missing)
        pg-0084.txt    (missing)
        pg-1342.txt    (missing)
        pg-1661.txt    (missing)
        pg-2701.txt    (missing)
    provenance/
        pg-0011.json   (missing)
        pg-0055.json   (missing)
        pg-0084.json   (missing)
        pg-1342.json   (missing)
        pg-1661.json   (missing)
        pg-2701.json   (missing)
```

## Missing filenames

| Work | Source filename | Provenance filename |
|------|-----------------|---------------------|
| PG-11 Alice's Adventures in Wonderland | `sources/pg-0011.txt` | `provenance/pg-0011.json` |
| PG-55 The Wonderful Wizard of Oz | `sources/pg-0055.txt` | `provenance/pg-0055.json` |
| PG-84 Frankenstein | `sources/pg-0084.txt` | `provenance/pg-0084.json` |
| PG-1342 Pride and Prejudice | `sources/pg-1342.txt` | `provenance/pg-1342.json` |
| PG-1661 The Adventures of Sherlock Holmes | `sources/pg-1661.txt` | `provenance/pg-1661.json` |
| PG-2701 Moby-Dick | `sources/pg-2701.txt` | `provenance/pg-2701.json` |

## Provenance sidecar template

Copy `research-data/incoming/AEON-LBC-1/provenance/PROVENANCE_TEMPLATE.json`
and fill in:

```json
{
  "schema_version": 1,
  "ebook_id": 11,
  "title": "Alice's Adventures in Wonderland",
  "source_provider": "Project Gutenberg",
  "source_page": "https://www.gutenberg.org/ebooks/11",
  "retrieval_date": "YYYY-MM-DD",
  "retrieval_method": "manual_browser_download",
  "format": "plain_text_utf8",
  "public_domain_basis": "public_domain_in_usa",
  "provided_by": "repository_owner"
}
```

All ten fields are required. `retrieval_method` must be one of
`manual_browser_download`, `manual_official_download`, or
`authorized_offline_transfer`. `format` must be `plain_text_utf8`.
`public_domain_basis` must be `public_domain_in_usa` or
`public_domain`. Personal identifiers are not required — the
`provided_by` field accepts a role label such as
`repository_owner`.

## Exact resume command

Once every file above is present:

```bash
python scripts/vendor_aeon_lbc1.py \
  --source-dir research-data/incoming/AEON-LBC-1 \
  --package-root research-data/AEON-LBC-1
```

If any source or provenance sidecar fails validation, no package
state changes — the previous `research-data/AEON-LBC-1/` (if any)
remains authoritative. Rerun after the failing input is corrected.

## Current clean commit and regression count

- Branch: `claude/funny-cori-a3k5cf`
- Head after this commit: pushed to origin.
- Working tree clean at push time.
- Full regression: **513/513 checks passing** (was 494; +19 from
  the offline-import suite).
- `docs/latent_bypass/status.json.achieved_claim_level = 0` — unchanged.
- `docs/latent_bypass/status.json.real_corpus_claims_authorized = false` — unchanged.
- IP-preservation firewall: PASS.
- Architecture-preservation firewall: PASS.

## What DID land in this tranche

- `scripts/vendor_aeon_lbc1.py::import_offline_sources` — reads
  local files under `<intake>/sources/`, validates via strict UTF-8
  decode, HTML/PDF/binary sniff, Gutenberg-marker check, work-
  specific title-evidence check (versioned per work), and per-file
  provenance sidecar validation. Every source is validated in
  memory BEFORE any byte is copied into the package; atomic
  promotion via `.tmp` + `os.replace` and post-copy digest
  reverification. Acquisition method recorded as
  `manual_official_download` with
  `executing_environment_did_not_download=true`.
- `scripts/vendor_aeon_lbc1.py` CLI grows `--source-dir` (mutually
  exclusive with `--refresh-source` and `--ca-bundle`).
- `research-data/incoming/AEON-LBC-1/README.md` and
  `provenance/PROVENANCE_TEMPLATE.json` document the intake
  contract.
- `tests/test_aeon_lbc1_offline_import.py` (+19): mutual
  exclusion, exactly-six-required, extras rejected, missing
  rejected, Gutenberg marker required, wrong-work content
  rejected, HTML masquerade rejected, PDF masquerade rejected,
  binary data rejected, invalid UTF-8 rejected, source digest
  binds downstream artifacts, missing provenance rejected, invalid
  `retrieval_method` rejected, `ebook_id` mismatch rejected,
  intake files remain byte-identical after import, partial import
  never promotes over the previous package, corpus notices do not
  touch Aeon license, no corpus enters the installer bundle,
  `--refresh-source` refused in offline mode.

## Preserving Aeon IP

The manually supplied corpus is input data only. It does not alter
the ownership or license of Aeon source, architecture, weights,
checkpoints, algorithms, evidence, runtime, or installer. Corpus
licence text stays inside the corpus package.

No Aeon artifact is sent externally during corpus intake — the
importer is offline by construction and holds no network client
handle in the intake code path.
