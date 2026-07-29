# F2 — Artifact Identity and Provenance

**Source of truth:** `aeon/provenance.py`, `aeon/corpus_manifest.py`, `docs/tcb_report.json`, `docs/corpus_manifest_schema.json`.
**Enforcement:** `tests/test_provenance.py` (14 checks).

## 1. Canonical identity (§F2.1)

`aeon/provenance.py::canonicalize` + `hash_object` produce environment-portable identities. Rules:

- Dict keys sorted; environment-incidental keys (`absolute_path`, `hostname`, `tmpdir`, `cwd`, `run_started_at`, `wall_clock`, `epoch`, `container_id`, `user`) stripped before hashing.
- Lists retain order (order IS semantic).
- Canonical JSON serialisation: `sort_keys=True, separators=(",", ":"), ensure_ascii=False`.

**Result:** the same semantic content produces the same sha256 regardless of machine, path, temp dir, or timestamp — tested by `test_canonical_json_ignores_env_incidentals` and `test_canonical_json_key_order_agnostic`.

## 2. Provenance chain (§F2.2)

`Source → Build → Configuration → Tokenizer → Corpus → TrainingRun → Checkpoint → Evaluation → Recovery`

Encoded as `aeon/provenance.py::CHAIN_KINDS` + `ProvenanceRecord`. A `checkpoint` record MUST reference:

- source_commit (with `dirty` flag)
- build_configuration (dependency lockfile sha + runtime versions)
- model_configuration (semantic sha256 of the YAML config)
- tokenizer (sha256 + vocab_size + special-id layout)
- corpus_manifest (sha256 over the source manifest)
- training_run (run identity)
- runtime_policy (sha256 of the runtime policy)
- security_policy (sha256 of the security policy)
- patch_manifest_version (six-patch identity)
- architecture_manifest_version

**Refusal:** `strict_verify(rec, kind="checkpoint")` raises `ProvenanceError` when any required field is missing or when `source_commit` is unknown (§F2.5).

## 3. Dependency policy (§F2.3)

`docs/tcb_report.json` enumerates every runtime dependency (`torch`, `safetensors`, `sentencepiece`, `pyyaml`, `numpy`), the version pin, why it is required, its integrity posture, and where in `aeon/` it is used.

- **Explicitly declared:** in `pyproject.toml`.
- **Version-pinned:** yes (see TCB report).
- **Loaded only from approved locations:** operator-provisioned index, no runtime `pip install` — verified by `test_no_runtime_pip_call_in_forward_path` (scans `aeon/` and `scripts/`).
- **Absent from automatic model-controlled installation paths:** `install_policy.auto_install_from_aeon_runtime` is `false`.

Removed dependencies (previously in the tree, now confirmed absent): `transformers`, `accelerate`, `huggingface_hub`, `datasets`.

## 4. Corpus provenance (§F2.4)

Schema in `docs/corpus_manifest_schema.json`, validator in `aeon/corpus_manifest.py::validate_manifest`. Required fields per source: `source_id`, `origin`, `acquired_at`, `license_status`, `content_sha256`, `preprocessing_version`, `filtering_version`, `deduplication_version`, `partition_assignment`, `inclusion_status`, `rejection_reason_if_rejected`, `trust_level`.

**Refusal rules:**

- Missing any required field → `ProvenanceError`.
- `trust_level=quarantined` cannot enter `train` / `validation` partitions.
- `inclusion_status=excluded` MUST provide `rejection_reason_if_rejected`.
- `content_sha256` recomputation mismatch → `ProvenanceError` (`verify_source_content`).

The **synthetic-token smoke path** used when no corpus is configured records itself as `source_id=synthetic_random_tokens`, `license_status=not_applicable_synthetic`. This preserves the invariant that unidentified data cannot enter the certified path — the synthetic case is *identified*.

## 5. Refusal behaviour (§F2.5)

`aeon/provenance.py::strict_verify` refuses:

- Missing provenance fields for a `checkpoint`, `recovery`, or `eval` artefact.
- Unknown source commit.
- Tokenizer marked `present=True` with no `sha256`.

`aeon/corpus_manifest.py::refuse_if_invalid` refuses invalid manifests (raising `ProvenanceError` with a concise concatenation of the first errors).

## Exit gate

- [x] Artefact identities are canonical and portable (environment-incidental keys stripped).
- [x] Provenance chain covers Source → Build → Config → Tokenizer → Corpus → TrainingRun → Checkpoint → Evaluation → Recovery.
- [x] Invalid or incomplete provenance fails closed (`strict_verify`, `refuse_if_invalid`).
- [x] Corpus identity is reproducible (source `content_sha256` + preprocess/filter/dedup versions).
- [x] Absolute machine paths do not affect semantic artefact identity (test).
- [x] Inherited 69-check suite + 14 new F2 checks = 83/83.
