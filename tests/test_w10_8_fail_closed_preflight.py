"""W10-8 — fail-closed frozen preflight.

Audit finding A17: run_preflight returns READY_WITH_WARNINGS (not BLOCKED)
when the tokenizer or corpus is missing or unusable. In the frozen Windows
build that reads as "the desktop shows Ready" while the worker would fall
through to torch.randint synthetic tokens on Start.

W10-8 corrects the frozen path only:
    * frozen mode + no tokenizer_path                    -> BLOCKED
    * frozen mode + tokenizer_path missing on disk       -> BLOCKED
    * frozen mode + tokenizer_path unloadable            -> BLOCKED
    * frozen mode + no corpus_path                       -> BLOCKED
    * frozen mode + corpus_path missing on disk          -> BLOCKED
    * frozen mode + corpus yields zero records           -> BLOCKED
    * source-tree mode preserves the READY_WITH_WARNINGS behaviour so
      developers can iterate without a corpus.
"""
import json
import os
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _run_preflight(cfg, frozen: bool):
    """Run run_preflight with a controlled 'is this a frozen build?' verdict."""
    from aeon.config import preflight
    with mock.patch.object(preflight, "_is_frozen", return_value=frozen):
        return preflight.run_preflight(cfg)


# ---------------------------------------------------------------------------
def test_frozen_mode_blocks_when_no_tokenizer_configured():
    r = _run_preflight({}, frozen=True)
    assert r.verdict.value == "BLOCKED", r.as_dict()
    tok = [c for c in r.checks if c.name == "tokenizer"][0]
    assert tok.status == "fail", tok
    assert "frozen mode requires a tokenizer" in tok.detail.lower()


def test_source_mode_still_warns_when_no_tokenizer_configured():
    r = _run_preflight({}, frozen=False)
    tok = [c for c in r.checks if c.name == "tokenizer"][0]
    assert tok.status == "warn", tok


def test_frozen_mode_blocks_when_no_corpus_configured():
    r = _run_preflight({}, frozen=True)
    corpus = [c for c in r.checks if c.name == "corpus"][0]
    assert corpus.status == "fail", corpus


def test_frozen_mode_blocks_when_corpus_yields_zero_records():
    with tempfile.TemporaryDirectory() as d:
        empty = Path(d) / "empty.jsonl"
        empty.write_text("")
        # Also a tokenizer stub so the tokenizer check doesn't dominate
        r = _run_preflight({"corpus_path": str(empty)}, frozen=True)
        corpus = [c for c in r.checks if c.name == "corpus"][0]
        assert corpus.status == "fail", corpus
        assert "unusable" in corpus.detail.lower() or "zero" in corpus.detail.lower()


def test_frozen_mode_blocks_when_tokenizer_missing_on_disk():
    r = _run_preflight({"tokenizer_path": "/definitely/does/not/exist.model"},
                        frozen=True)
    tok = [c for c in r.checks if c.name == "tokenizer"][0]
    assert tok.status == "fail", tok


def test_frozen_mode_blocks_when_corpus_missing_on_disk():
    r = _run_preflight({"corpus_path": "/definitely/does/not/exist.jsonl"},
                        frozen=True)
    corpus = [c for c in r.checks if c.name == "corpus"][0]
    assert corpus.status == "fail", corpus


def test_frozen_mode_passes_when_tokenizer_and_corpus_are_real():
    """Sanity: with a real Aeon tokenizer trained on a tiny fixture and a
    real .jsonl corpus, the frozen path should not spuriously BLOCK."""
    import sentencepiece as spm
    with tempfile.TemporaryDirectory() as d:
        # Tokenizer (many short sentences to satisfy sentencepiece)
        raw = Path(d) / "raw.txt"
        sentences = ["the quick brown fox jumps over the lazy dog",
                     "a small red car parked on the street",
                     "she opened the door and walked outside",
                     "he laughed and closed the book slowly",
                     "birds sing at dawn on the old oak tree",
                     "the ship sailed across the deep blue sea",
                     "children played in the sunny green park",
                     "we watched the clouds drift over the hill"]
        raw.write_text("\n".join(sentences * 40))
        model_prefix = str(Path(d) / "tok")
        spm.SentencePieceTrainer.train(
            input=str(raw), model_prefix=model_prefix, vocab_size=200,
            character_coverage=0.9995, model_type="bpe",
            bos_id=1, eos_id=2, pad_id=0, unk_id=3)
        # Corpus
        corpus = Path(d) / "c.jsonl"
        text_body = " ".join(sentences)
        corpus.write_text(json.dumps({"text": text_body}) + "\n"
                          + json.dumps({"text": text_body}) + "\n")
        r = _run_preflight({
            "tokenizer_path": model_prefix + ".model",
            "corpus_path": str(corpus),
        }, frozen=True)
        tok = [c for c in r.checks if c.name == "tokenizer"][0]
        corpus_c = [c for c in r.checks if c.name == "corpus"][0]
        assert tok.status == "pass", tok
        assert corpus_c.status == "pass", corpus_c


def test_source_level_check_preflight_has_frozen_branch():
    src = open(os.path.join(ROOT, "aeon/config/preflight.py"), encoding="utf-8").read()
    assert "def _is_frozen" in src, "preflight must gate BLOCKED on frozen mode"
    assert "unusable_status" in src, (
        "preflight must select fail-vs-warn per frozen mode")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
