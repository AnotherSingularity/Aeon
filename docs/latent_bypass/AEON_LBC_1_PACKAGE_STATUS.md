# AEON-LBC-1 corpus package — status and reproduction instructions

**Status**: NOT VENDORED. Every downstream L-tranche (L3, L4, L5) that
depends on the real corpus is BLOCKED until the six Project Gutenberg
UTF-8 plain-text sources land under
`research-data/AEON-LBC-1/source/`.

The blocker is the environment's outbound network policy (the egress
proxy answers `403 Forbidden` on `CONNECT www.gutenberg.org:443` and
every documented mirror). Runtime, training, validation, diagnosis,
and evaluation remain offline as required.

## Six required source files

| eBook | Work | Partition role | Vendor filename |
|------:|------|----------------|-----------------|
| PG-2701 | Moby-Dick; Or, The Whale | train | `pg-2701-moby-dick.txt` |
| PG-1342 | Pride and Prejudice | train | `pg-1342-pride-and-prejudice.txt` |
| PG-11 | Alice's Adventures in Wonderland | train | `pg-0011-alice.txt` |
| PG-84 | Frankenstein; Or, The Modern Prometheus | calibration | `pg-0084-frankenstein.txt` |
| PG-55 | The Wonderful Wizard of Oz | validation | `pg-0055-wizard-of-oz.txt` |
| PG-1661 | The Adventures of Sherlock Holmes | test (sealed) | `pg-1661-sherlock-holmes.txt` |

Every URL, the SHA-256 record, and the exact one-time acquisition
policy live in `scripts/vendor_aeon_lbc1.py`. The script:

1. Only pulls the six eBook IDs above.
2. Refuses non-HTTPS, non-`*.gutenberg.org`, non-`text/plain` responses.
3. Refuses redirects outside `gutenberg.org`.
4. Enforces a 25 MiB per-file response ceiling and a 30 s timeout.
5. Records retrieval timestamp (UTC) and the resolved URL.
6. Preserves the complete original file byte-for-byte under
   `source/` and its SHA-256 in `ORIGINAL_SOURCE_DIGESTS`.
7. Refuses to overwrite an existing source whose recorded digest
   differs from the on-disk digest unless `--refresh-source` is passed.

## Reproduction commands (once egress policy allows Gutenberg)

```bash
# 1. Vendor the six sources into research-data/AEON-LBC-1/source/
python scripts/vendor_aeon_lbc1.py --package-root research-data/AEON-LBC-1

# 2. Deterministic preprocessing to research-data/AEON-LBC-1/processed/
#    (aeon-lbc1-v1). Fails-closed on missing header/footer markers.
python scripts/prepare_aeon_lbc1.py --package-root research-data/AEON-LBC-1

# 3. Emit tokenizer_binding.json, partition_manifest.json,
#    leakage_report.json, sealed_test_manifest.json, PACKAGE_MANIFEST.json.
#    (helper script; see below.)
python scripts/build_aeon_lbc1_manifests.py --package-root research-data/AEON-LBC-1

# 4. Validate the assembled package. Refuses when anything is missing.
python -c "from aeon.bypass.corpus_package import validate_corpus_package; \
           r = validate_corpus_package('research-data/AEON-LBC-1'); \
           print(r)"

# 5. Once ready_for_L3 == True, the runner may begin the P0/P1/P2 stages.
python scripts/run_aeon_lbc1_stage.py --stage P0 --config configs/latent_bypass/aeon_lbc1_proxy.yaml
python scripts/run_aeon_lbc1_stage.py --stage P1 --config configs/latent_bypass/aeon_lbc1_proxy.yaml
python scripts/run_aeon_lbc1_stage.py --stage P2 --config configs/latent_bypass/aeon_lbc1_proxy.yaml
```

The manifest-builder / runner scripts land alongside the corpus in a
follow-on tranche once the sources are on disk. They are NOT part of
the runtime; they never run inside `Aeon.exe`.

## Sealed-test control

Once the P2 checkpoint is produced and thresholds and the reaction
coordinate are locked, the operator commits
`docs/latent_bypass/L3_CALIBRATION_LOCK.json` with the twelve required
fields (see `aeon/bypass/sealed_partition.py::LOCK_ARTIFACT_REQUIRED_KEYS`).
Only after that artifact validates does
`aeon.bypass.sealed_partition.read_sealed_partition()` yield test
records. Any change to any input to the lock artifact forces a new
`experimental_version` string; previously opened test results are not
fresh confirmatory evidence.

## Why this file exists

Program B refuses to fabricate scientific evidence against the
synthetic-English fixture. The infrastructure needed to vendor,
preprocess, partition, validate, seal, tokenize, and train against
the real corpus is complete and unit-tested (43 tests across
`test_aeon_lbc1_acquisition.py`, `test_l3_reaction_coordinate.py`,
`test_l4_telemetry_l5_interventions.py`). The one missing input is
the corpus itself, and its acquisition is blocked at the environment
level, not the code level. See `docs/latent_bypass/L2_TO_L3_STATE_B.md`
for the STATE B report the directive requires.
