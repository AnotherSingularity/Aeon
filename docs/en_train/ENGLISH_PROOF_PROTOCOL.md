# ENGLISH_PROOF_PROTOCOL

**Status.** Infrastructure landed. **Halted at
`AWAITING_DOLLY_DATA_UPLOAD`.** The authorized `databricks/databricks-dolly-15k`
corpus has not been retrieved because the sandbox's network policy
denies `huggingface.co` (proxy returns 403 CONNECT). No pilot has run.
No candidate has been produced. No English-quality claim is made.

**Scope.** Prove that Aeon's native 7M architecture can learn to
produce normal English sentences from human-authored instruction /
response data, before performing another Windows build.

**Corpus.** Exactly `databricks/databricks-dolly-15k`. CC BY-SA 3.0.
No substitution permitted. No LLM-generated data, teacher logits,
distillation, judge model, grammar corrector, retrieval-generated
answers, canned responses, response rewriting, post-generation
repair, LoRA, adapters, new architectural modules, fallback language
model, API inference, or online parameter updates.

## Boundaries (invariants preserved by the pilot)

Everything below is unchanged by this tranche and is asserted to
remain unchanged by the pilot:

| Item | Value |
| ---- | ----- |
| A₀ digest | `sha256:2f895a05411567619371dd76a5f22868ca9e7edc17f33711e2e99aab04a972f9` |
| `total_parameters` | 7,015,366 |
| `state_dict_key_count` | 67 |
| `K` | 16 |
| `MARGIN_H` | 0.02 |
| `MARGIN_C` | 0.02 |
| Protected P2 SHA-256 | `sha256:962fcd5e65a88e3b6d061e73968bea7eb3581c4d8a15b49be82427004db9fc3c` |
| Tokenizer SHA-256 | `sha256:064ab6a98ee4b8177249f14dc09e60ccaa9986b66cb84a4072c68dc4de533481` |
| Renderer | `aeon/desktop/runtime.py::_generate` — `D_stream(y) == D_full(y)` |
| Fast-clock cadence | one `substrate.step` per token (`aeon/hybrid.py:154-158`) |
| Slow-clock cadence | one `recursion.step` per K-window (`aeon/hybrid.py:175-177`) |

Live-captured pre-training baseline is recorded in
`docs/en_train/english_proof_invariance.json` (67-entry state-dict
manifest, live A₀ digest, live margins, both protected SHA-256s
matching pinned).

## Data flow

1. **Provenance.** `docs/en_train/dolly15k_provenance.json` — schema
   template. All nullable acquisition fields are `null` in this
   session; the operator populates them from the actual upload.
2. **Acquisition.** Operator places the raw Dolly-15k artifact under
   `research-data/incoming/EN-DOLLY-15K/sources/` and records:
   * canonical URL
   * resolved revision
   * retrieval timestamp
   * byte size
   * SHA-256
   * license (CC BY-SA 3.0)
   * attribution requirement
   * official human-authorship statement
   * raw record count, category counts, fields, null rates
3. **Split.** `aeon.en_train.dolly_split.deterministic_split`:
   90% train / 5% validation / 5% sealed test, deterministic grouped
   with 5-gram Jaccard near-duplicate clustering at threshold 0.85.
   Near-duplicates stay in the same partition. Exact duplicates are
   collapsed to the lexicographic representative and excluded records
   are recorded with `(record_id, reason)`. Manifest is written to
   `docs/en_train/dolly15k_split_manifest.json` and includes a
   `sealed_test_lock_sha256` that any later mutation breaks.
4. **Serialization.** `aeon.en_train.proof_pilot.render_dolly_record_for_training`
   uses the same `user: ...\n\nassistant: ...` prompt/response contract
   as the desktop runtime. The assistant character-span identifies
   the SUPERVISED region for the response mask.

## Training mathematics (unchanged Aeon architecture)

Response-masked causal cross-entropy over Aeon's native logits, per
the directive Section 7:

```
L(θ) = − (1 / Σ_{j,t} m_{j,t}) · Σ_{j,t} m_{j,t} · log p_θ(s_{j,t} | s_{j,<t})
```

with `m_{j,t} = 1` iff token `s_{j,t}` belongs to the human response.

Optimizer step (Aeon's existing approved path,
`aeon.en_train.trainer.train_one_step`):

```
θ_{k+1} = G_existing(θ_k, ∇_θ L_k)
```

`k` is an offline optimizer-step index — **not** Aeon's fast clock,
**not** Aeon's slow clock. During inference `θ^{(n+1)} = θ^{(n)}`
(witnessed by `tests/test_desktop_inference_immutability.py`).

## Bounds and fail-closed gates

* Hard maximum: **3,000,000 response-training tokens.**
* Checkpoint targets near 250K, 500K, 1M, 2M, 3M tokens.
* Candidate selected via validation loss + unsealed dev probes only.
* Sealed test evaluated **exactly once** after candidate selection.
* Immediate stop on: NaN / Inf loss · gradient explosion · certificate
  failure · architecture drift · parameter-count drift · tokenizer
  drift · P2 mutation · unexpected state-dict change · recurrence /
  clock-test failure.

## Weight-only attribution

`aeon.en_train.proof_harness.AttributionSettings` freezes:
context_length, max_new_tokens, greedy/temperature/top_k/top_p,
repetition_penalty, stop_on_eos, deterministic_seed,
prompt_serialization_id, renderer_id. The fingerprint compares
bytewise; `assert_attribution_settings_bytewise_equal(a, b)` fails
loudly on any drift. P2 and the candidate must produce every
proof response under bytewise-identical settings; θ is the only
permitted difference.

Per response, the harness records: prompt id, prompt text, checkpoint
role, checkpoint SHA-256, generated token ids, per-step selected
token, full decoded text, streamed decoded text, stop reason, settings
fingerprint, and generation duration.

## Human proof gate

The candidate is **not** approved by any automated check. The pilot
produces a blinded scorecard of 25 sealed prompts (see
`docs/en_train/english_proof_blind_scorecard.csv` after the pilot).
Response A vs response B is randomised per prompt; the hidden A→P2 /
B→candidate mapping is stored under
`docs/en_train/english_proof_blind_mapping.json` (or a hashed record
of same). Dylan marks per prompt:

* complete grammatical sentence: yes/no
* relevant to the instruction: yes/no
* understandable without guessing: yes/no
* whaling contamination: yes/no
* preferred response: A/B/tie
* notes

Provisional pass gate:

```
complete readable sentence: ≥ 20/25
relevant response:          ≥ 18/25
understandable response:    ≥ 18/25
whaling contamination:      ≤ 1/25
joined-word renderer defect: 0/25
```

The candidate is not approved until Dylan scores the sheet.

## Live-source demo command (once a candidate exists)

```bash
python -m aeon.entry --chat \
    --release-root release-assets/aeon-desktop-p2-proxy \
    --candidate-weights runs/en_proof_dolly15k_s20260822/AEON-EN-PROOF-DOLLY15K-S20260822/selected.pt \
    --banner "ENGLISH PROOF CANDIDATE — NOT RELEASE APPROVED"
```

Uses the selected candidate weights, the protected tokenizer, the
corrected renderer, and offline execution. The production release
bundle and P2 release id are unchanged.

## Halt states

* `AWAITING_DOLLY_DATA_UPLOAD` — corpus not uploaded, no substitute
  used. **Current state at head `e602ba9` (this commit will be
  updated as EN-PROOF-C lands).**
* `ENGLISH_PROOF_FAILED_NO_PACKAGING` — a fail-closed gate tripped.
* `ENGLISH_PROOF_READY_FOR_DYLAN_REVIEW` — pilot completed within
  bounds, all invariants held, evidence written, blinded scorecard
  produced; **only Dylan's human score decides whether the candidate
  is approved.**

Under every outcome:

```
NO WINDOWS PACKAGING EXECUTED
```
