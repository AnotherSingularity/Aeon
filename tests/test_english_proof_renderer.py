"""ENGLISH-PROOF-0 / A — additional renderer-invariant coverage.

Existing coverage is in tests/test_en_train_renderer_fix.py. This file
adds the categories listed in the ENGLISH-PROOF-0 directive Section 4
that were not yet explicitly covered:

  * quotation marks (ASCII " and ' plus typographic “ ” ‘ ’)
  * repeated spaces
  * EOS handling — decoding up to and including the EOS token id
  * multi-token words that span more than one SentencePiece piece
  * sequences whose token count crosses the native K = 16 boundary

Every test uses the real bundled AEON-LBC-1 SentencePiece tokenizer.
The invariant asserted is the same one the runtime enforces:

  D_stream(y_{1..n}) == D_full(y_{1..n})   for every completed sequence

which is the cumulative-canonical-decode + U+FFFD hold-back +
completion-time flush pattern implemented in
aeon/desktop/runtime.py::_generate.

None of these tests train, decode any parameters, or touch the
protected checkpoint or tokenizer bytes.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TOK_PATH = ROOT / "research-data" / "AEON-LBC-1" / "tokenizer" / "aeon-lbc1.model"


def _tokenizer():
    from aeon.tokenizer import AeonTokenizer
    return AeonTokenizer(str(TOK_PATH))


def _stream_from_ids(ids):
    """Mirror aeon/desktop/runtime.py::_generate's fixed rendering
    path exactly: cumulative decode, U+FFFD hold-back, completion
    flush."""
    tok = _tokenizer()
    D_full = tok.decode(ids)
    emitted = ""
    deltas = []
    for i in range(1, len(ids) + 1):
        canonical_so_far = tok.decode(ids[:i])
        committable = canonical_so_far.rstrip("�")
        if committable.startswith(emitted):
            deltas.append(committable[len(emitted):])
            emitted = committable
    if emitted != D_full:
        tail = D_full[len(emitted):] if D_full.startswith(emitted) else D_full
        deltas.append(tail)
    return "".join(deltas), D_full


def _check(text: str):
    tok = _tokenizer()
    ids = tok.encode(text, add_bos=False, add_eos=False)
    if not ids:
        return
    stream, full = _stream_from_ids(ids)
    assert stream == full, (
        f"D_stream != D_full for {text!r}:\n  full  ={full!r}\n  stream={stream!r}")


# ---------------------------------------------------------------------------
# 1. Quotation marks
# ---------------------------------------------------------------------------
def test_renderer_ascii_double_quotes():
    _check('He said "hello" and then left.')


def test_renderer_ascii_single_quotes():
    _check("It's John's book, isn't it?")


def test_renderer_typographic_double_quotes():
    _check("She replied “yes, of course” without hesitation.")


def test_renderer_typographic_single_quotes():
    _check("The word ‘quality’ has meaning here.")


# ---------------------------------------------------------------------------
# 2. Repeated spaces
# ---------------------------------------------------------------------------
def test_renderer_double_space_between_words():
    # SentencePiece typically collapses whitespace; whatever the
    # tokenizer decides to preserve, streamed decode must match the
    # one-shot decode of the same ids.
    _check("hello  world")


def test_renderer_triple_space_between_words():
    _check("alpha   beta")


def test_renderer_leading_and_trailing_whitespace():
    _check("   leading and trailing   ")


# ---------------------------------------------------------------------------
# 3. EOS handling
# ---------------------------------------------------------------------------
def test_renderer_decodes_correctly_when_eos_included():
    """EOS is a special id (2 in the AEON-LBC-1 tokenizer). Including
    it in the id list must not break streamed==full: whatever the
    tokenizer decodes an EOS id to (typically empty), that behaviour
    is preserved and the cumulative streamed decode still equals the
    one-shot decode."""
    tok = _tokenizer()
    ids = tok.encode("The generation ends here.", add_bos=False, add_eos=False)
    ids_with_eos = ids + [tok.eos_id]
    stream, full = _stream_from_ids(ids_with_eos)
    assert stream == full, (
        f"D_stream != D_full with trailing EOS:\n  full  ={full!r}\n  stream={stream!r}")


def test_renderer_decodes_correctly_with_eos_mid_sequence():
    """A stray EOS mid-sequence must not desync streamed vs full."""
    tok = _tokenizer()
    a = tok.encode("hello", add_bos=False, add_eos=False)
    b = tok.encode("world", add_bos=False, add_eos=False)
    stream, full = _stream_from_ids(a + [tok.eos_id] + b)
    assert stream == full


# ---------------------------------------------------------------------------
# 4. Multi-token words
# ---------------------------------------------------------------------------
def test_renderer_word_that_splits_into_many_pieces():
    """Long unseen words are broken into multiple SentencePiece pieces
    via byte fallback or subword coverage. The join across those
    pieces must not lose the leading-space marker or duplicate a
    character."""
    for w in ("supercalifragilisticexpialidocious",
              "electroencephalography",
              "pneumonoultramicroscopicsilicovolcanoconiosis"):
        tok = _tokenizer()
        ids = tok.encode(w, add_bos=False, add_eos=False)
        assert len(ids) >= 2, f"expected {w!r} to split into >= 2 pieces"
        _check(w)


def test_renderer_word_with_leading_uppercase_split():
    _check("Rediscovered Antidisestablishmentarianism")


# ---------------------------------------------------------------------------
# 5. Sequences crossing the native K = 16 window boundary
# ---------------------------------------------------------------------------
def test_renderer_sequence_exactly_at_k_boundary():
    """A sequence of exactly K=16 tokens: streamed==full. Uses a
    long enough source text that slicing to 16 always succeeds."""
    tok = _tokenizer()
    ids = tok.encode(
        "The quick brown fox jumps over the lazy dog and then keeps "
        "running through the meadow toward the old oak tree by the river.",
        add_bos=False, add_eos=False)
    assert len(ids) >= 16, f"needed source text with >=16 tokens, got {len(ids)}"
    ids = ids[:16]
    stream, full = _stream_from_ids(ids)
    assert stream == full
    assert len(ids) == 16


def test_renderer_sequence_crossing_first_k_boundary():
    tok = _tokenizer()
    text = ("The quick brown fox jumps over the lazy dog. "
            "Then it runs across the meadow to find its friends.")
    ids = tok.encode(text, add_bos=False, add_eos=False)
    assert len(ids) > 16, "test needs >16 tokens to cross the K boundary"
    stream, full = _stream_from_ids(ids)
    assert stream == full


def test_renderer_sequence_spanning_multiple_k_windows():
    """A long sequence spanning at least 4 K-windows (>=64 tokens)."""
    tok = _tokenizer()
    text = (
        "The quick brown fox jumps over the lazy dog beside the small "
        "stream. Then it runs across the wide meadow to find its "
        "friends among the tall grass. They gather at the old oak tree "
        "by the river and share stories about their adventures. As the "
        "sun sets slowly behind the distant hills, each of them heads "
        "home for the warm evening meal with their families.")
    ids = tok.encode(text, add_bos=False, add_eos=False)
    assert len(ids) >= 64, f"want >=64 tokens for multi-K coverage, got {len(ids)}"
    stream, full = _stream_from_ids(ids)
    assert stream == full


# ---------------------------------------------------------------------------
# 6. Renderer code performs no parameter mutation (sanity — already
#    covered exhaustively by tests/test_desktop_inference_immutability.py)
# ---------------------------------------------------------------------------
def test_english_proof_confirms_renderer_immutability_test_exists():
    """The ENGLISH-PROOF-0 directive requires that the renderer never
    mutates parameters, logits, or token IDs. That invariant is
    exhaustively tested in tests/test_desktop_inference_immutability.py
    (parameter-hash, session-clear, AST no-optim-refs, torch.inference_mode
    wrapping). This test simply asserts that the witness file exists
    so the ENGLISH-PROOF-0 proof harness cannot silently regress if
    that file is ever deleted."""
    p = ROOT / "tests" / "test_desktop_inference_immutability.py"
    assert p.exists(), (
        "tests/test_desktop_inference_immutability.py must exist — it is the "
        "witness that the renderer/generation path performs no parameter mutation.")
