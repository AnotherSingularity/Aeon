# Renderer Fix Proof (AFTER)

**Verdict:** `D_stream(y) = D_full(y)` for every fixture. Token IDs
byte-identical to the defect-proof run. Weights, tokenizer, config,
architecture fingerprint — all unchanged.

Machine-readable: `docs/en_train/renderer_fix_proof.json`.
Test suite: `tests/test_en_train_renderer_fix.py` (16 checks).

---

## What changed

Only `aeon/desktop/runtime.py::AeonDesktopRuntime._generate`, at the
per-step `TEXT_DELTA` emission site and a new completion-time flush.

Before (defective):
```python
text_delta = tok.decode(generated_ids[-1:])   # per-token; drops ▁
```

After (canonical decode + tail-only delta + byte-fallback hold-back +
completion flush):
```python
canonical_so_far = tok.decode(generated_ids)   # cumulative
committable = canonical_so_far.rstrip("�") # hold back incomplete UTF-8 tail
if committable.startswith(emitted_text):
    text_delta = committable[len(emitted_text):]
    emitted_text = committable
else:
    text_delta = ""                              # mojibake replaced mid-stream
# ...at completion:
if _final_full != emitted_text:
    self._emit(EventKind.TEXT_DELTA, ..., payload={"delta": tail, "flush": True})
```

No other file in `aeon/**` was modified.

---

## Fixture results (deterministic greedy, same 12-token generations
as `docs/en_train/renderer_defect_proof.json`)

| Prompt              | D_full (canonical)                                    | D_stream (fixed)                                     | Match | Token IDs unchanged |
| ------------------- | ----------------------------------------------------- | ---------------------------------------------------- | ----- | ------------------- |
| `The`               | `the whale, and the whale, and the whale, and`        | `the whale, and the whale, and the whale, and`       | ✔     | ✔                   |
| `Once upon a time`  | `, and the whale, and the whale, and the whale`       | `, and the whale, and the whale, and the whale`      | ✔     | ✔                   |
| `Hello world`       | `, and the whale of the whale, and the whale of`      | `, and the whale of the whale, and the whale of`     | ✔     | ✔                   |

**Mismatch rate: 0/3.** **Token-ID drift rate: 0/3.**

The whaling fixation remains visible — that is a LANGUAGE
ACQUISITION problem for the training tranche, not the renderer.
Nothing about this fix conceals it.

---

## Isolation proofs

Every one of these is a passing test in `tests/test_en_train_renderer_fix.py`:

| Requirement                                             | Test                                                      |
| ------------------------------------------------------- | --------------------------------------------------------- |
| Model weights unchanged                                 | `test_renderer_fix_did_not_touch_p2_checkpoint` (sha256 match) |
| Tokenizer files unchanged                               | `test_renderer_fix_did_not_touch_tokenizer_bytes` (sha256 match) |
| Model configuration unchanged                           | `test_renderer_fix_did_not_touch_model_configuration` (sha256 match) |
| Architecture fingerprint A₀ unchanged                   | `test_renderer_fix_did_not_change_architecture_fingerprint_A0` |
| Selected token IDs unchanged for defect fixtures        | `test_renderer_fix_does_not_change_token_ids_for_defect_fixtures` |

`Δarchitecture(A₀, A_current) = 0` — proven by rebuilding the model
from the frozen config and comparing the freshly computed A₀ digest
to the one committed at
`docs/en_train/EN_TRAIN_ARCHITECTURE_FREEZE.json.architecture_fingerprint_A0_digest`.

---

## Coverage of §21 acceptance cases

| Case                                             | Test                                                            |
| ------------------------------------------------ | --------------------------------------------------------------- |
| Spacing between words                            | `test_renderer_spacing_between_words`                           |
| Punctuation                                       | `test_renderer_punctuation_and_ascii`                           |
| Contractions                                     | `test_renderer_contractions`                                    |
| Newlines / paragraphs                            | `test_renderer_newlines_and_paragraphs`                         |
| Latin-1 supplement Unicode                       | `test_renderer_unicode_common_latin1_supplement`                |
| Byte-fallback + extended Unicode (CJK / Greek)   | `test_renderer_unicode_extended_and_byte_fallback`              |
| Leading space after punctuation                  | `test_renderer_leading_space_after_punctuation`                 |
| Numeric + mixed                                  | `test_renderer_numeric_and_mixed`                               |
| Empty input                                      | `test_renderer_empty_string_is_no_op`                           |
| Live runtime: sum of deltas == full_text         | `test_live_runtime_join_of_deltas_equals_full_text`             |
| Live runtime: full_text == canonical decode      | `test_live_runtime_full_text_equals_canonical_decode`           |

---

## What this fix explicitly did NOT do

* Did not add missing words.
* Did not repair grammar.
* Did not change any token choice.
* Did not rewrite responses.
* Did not conceal repetition or the whaling fixation.
* Did not substitute a better answer.
* Did not consult any external model.
* Did not touch the model, tokenizer, config, checkpoint, or
  architecture fingerprint.

This is a **rendering correction**, not a **language-learning result**.
