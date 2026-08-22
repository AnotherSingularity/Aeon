# EN-TRAIN Corpus Intake Contract

**Status:** stopping gate — AWAITING_OFFLINE_CORPUS_SOURCES.

No training executes until authorized human-authored English sources
for `D_G`, `D_C`, `D_A`, and a human-authored sealed evaluation `D_E`
are dropped into the intake directories below and the validator
returns `passed_r_unk_gate: true`.

**Explicitly disallowed sources** (per the top-level directive):
LLM-generated text, teacher logits, synthetic assistant dialogue,
model-distilled data, retrieval-generated answers, or any prompt-
specific canned material.

**The existing six-book AEON-LBC-1 collection is baseline
provenance.** It must **not** be silently reused as the new
language-acquisition corpus. Any future reuse must respect §3's
per-book (≤0.5% of new training tokens) and per-author (≤2%)
caps — the current AEON-LBC-1 mix violates these because it is six
books total.

---

## 1. Directory layout (per intake `<CORPUS_ID>`)

```
research-data/incoming/<CORPUS_ID>/
    sources/
        <source_id>.<ext>              — raw human-written English
    provenance/
        <source_id>.json               — one per source
    licenses/
        <source_id>.txt (or LICENSE)   — license text or public-domain attestation
    manifests/
        intake.json                    — operator-supplied bundle description
```

Missing dir → `IntakeError(missing_intake_directory)`.
Source without matching provenance → `IntakeError(missing_provenance)`.

---

## 2. Required provenance fields

Every `provenance/<source_id>.json`:

```json
{
  "source_id": "<stable id used across all references>",
  "author_or_institution": "<name(s)>",
  "original_publication_location": "<URL, ISBN, or citation>",
  "acquisition_date": "YYYY-MM-DD",
  "license_or_public_domain_basis": "<license slug or 'public_domain_in_usa'>",
  "permitted_use_notes": "<any restrictions on training use>",
  "sha256": "sha256:<64 hex>",
  "byte_length": <int>,
  "document_count": <int>,
  "estimated_token_count": <int>,
  "encoding": "utf-8",
  "preprocessing_declaration": "<what will be applied — e.g. NFC + Gutenberg-wrapper strip + paragraph split>"
}
```

Missing field → `IntakeError(provenance_missing_fields)`.
`sha256` mismatch against source on disk → `IntakeError(source_digest_mismatch)`.

---

## 3. The four required intakes

### `D_G` — General English (Stage 1 target)

* Clean, diverse, human-written English.
* Enough distributional coverage for word boundaries, grammar,
  punctuation, ordinary vocabulary, and short + long dependencies.
* No single book > 0.5% of the training tokens; no single author > 2%.
* The existing AEON-LBC-1 six-book collection cannot satisfy this on
  its own — every book would be at ~16.7% of the mixture.
* Suggested intake id: `EN-GENERAL-01`.

### `D_C` — Conversational English (Stage 2 target)

* Human-written multi-turn conversations.
* Categories that must be represented: greetings, ordinary questions,
  concise answers, clarification, disagreement, uncertainty,
  multi-turn continuity, instruction comprehension, identity questions,
  natural conversational tone.
* Conversation-level partitioning (a conversation is one document —
  never split turns across partitions).
* Suggested intake id: `EN-CONVERSATIONAL-01`.

### `D_A` — Documented Aeon-specific statements

* Human-authored, Dylan-approved factual statements about Aeon
  (name, offline runtime, research-preview status, demonstrated
  capabilities).
* Never exceeds 2% of any Stage-2 batch.
* No unsupported marketing claims.
* Suggested intake id: `EN-AEON-IDENTITY-01`.

### `D_E` — Sealed evaluation

* At least 300 human-written prompts + deterministic scoring keys.
* Category minimums (§16):
  - 50 greetings
  - 50 factual-from-context
  - 50 one-part instructions
  - 50 two-part instructions
  - 40 uncertainty / insufficient-information
  - 40 two-turn continuity
  - 20 verified Aeon-identity
* No prompt or near-duplicate may enter training (validated by the
  Splitter's cross-partition duplicate check).
* No LLM authorship of `D_E` prompts or scoring keys.
* Suggested intake id: `EN-SEALED-EVAL-01`.

`SealedPrompt` records (one JSONL line each):

```json
{
  "prompt_id": "<stable id>",
  "category": "greetings|factual_from_context|instruction_one_part|instruction_two_part|uncertainty|continuity_two_turn|identity",
  "prompt_text": "<the prompt>",
  "context_turns": [{"role":"user","content":"…"}, …] | null,
  "scoring_key": {
    "required_contains": ["substr", …],
    "required_regex": ["pattern", …],
    "forbidden_contains": ["substr", …],
    "required_components": ["substr", …],
    "contradictory_components": ["substr", …],
    "min_chars": <int>,
    "max_chars": <int>
  }
}
```

---

## 4. Validator invocation

```
python scripts/en_train_validate_intake.py \
    --intake research-data/incoming/<CORPUS_ID>
```

Success signals:
* Layout OK (4 required dirs + one provenance per source).
* Every source's SHA-256 matches its declared provenance.
* Every source encodes strict UTF-8.
* `r_UNK ≤ 0.001` under the frozen tokenizer.

Failure surfaces:
* `missing_intake_directory` — add the missing dir.
* `missing_provenance` — add a matching JSON per source.
* `provenance_missing_fields` — see §2 above.
* `source_digest_mismatch` — the file on disk doesn't match the
  declared SHA-256; do NOT overwrite the file to fit the hash;
  reissue provenance if the source itself changed.
* `tokenizer_id_out_of_range` — a token ID escaped `[0, 16000)` (a
  broken tokenizer file or a sentence with byte sequences the
  16k vocab cannot cover — first re-check the tokenizer identity;
  do NOT silently retrain the tokenizer).

---

## 5. What the validator does NOT do

* Does not fetch, download, invent, synthesize, or generate any
  source content.
* Does not accept LLM-generated files.
* Does not deduplicate across `D_E` and any training partition
  automatically — the Splitter does that as it prepares partitions.
* Does not run training.

---

## 6. When ALL four intakes are ready

Training resumes with, per seed:

```
# Stage 1 (§7 mixture: 90% D_G / 10% D_C, D_A=0):
python scripts/en_train_run_stage.py --stage stage1 \
    --corpus-package research-data/<GENERAL_PACKAGE> \
    --tokenizer research-data/AEON-LBC-1/tokenizer/aeon-lbc1.model \
    --lr-pilot --seed 20260803 --out runs/en_train/stage1_s20260803/

# Stage 2 (§7 mixture: 35% D_G / 63% D_C / 2% D_A):
python scripts/en_train_run_stage.py --stage stage2 …

# Evaluation on each candidate against the sealed D_E:
python scripts/en_train_evaluate.py --checkpoint <candidate> \
    --sealed-eval <D_E path> --out docs/en_train/eval/<candidate>.json

# Attribution (§22):
python scripts/en_train_attribute.py \
    --p2 runs/aeon_lbc1_P2/final.pt --candidate <candidate>
```

Multi-seed reproduction of the final configuration follows §10 (single
seed for pilot / infrastructure validation) and §23 (three documented
seeds only after one configuration reaches the promotion thresholds
under a single seed). All three seeds are reported. No cherry-picking.
