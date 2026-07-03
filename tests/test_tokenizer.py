"""
Aeon tokenizer tests — the from-scratch tokenizer is Aeon's own, so it gets its
own conformance:

  1. train_tokenizer() produces a loadable model with the requested vocab size
  2. the fixed special-id layout holds (pad=0, unk=1, bos=2, eos=3)
  3. encode/decode round-trips text losslessly; add_bos/add_eos behave; decode
     strips control ids
  4. aeon/data.py reads .txt, .jsonl, and directories uniformly

Requires the `sentencepiece` backend; skips cleanly otherwise. Run:
    python tests/test_tokenizer.py     (or pytest)
"""
import itertools
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "scripts"))


def _have_spm():
    try:
        import sentencepiece  # noqa: F401
        return True
    except Exception:
        return False


def _varied_corpus(path):
    """Write a corpus with enough sub-word variety to train a small vocab."""
    subj = ["Aeon", "the substrate", "Recursion", "the transformer side", "the joiner",
            "the manifold", "the slow clock", "the certificate", "the readout", "the carry"]
    verb = ["integrates", "projects into", "reads", "writes", "holds", "advances",
            "contracts", "bounds", "conditions", "stabilizes", "aggregates", "injects"]
    obj = ["the contractive manifold", "a sigma below one", "the per-token signal",
           "the slow-clock window", "the recurrent state", "the certificate margin",
           "the write surface", "the token embeddings", "the spectral radius"]
    sents = [f"{s.capitalize()} {v} {o}." for s, v, o in itertools.product(subj, verb, obj)]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(sents) + "\n")


def test_train_and_roundtrip():
    if not _have_spm():
        print("  [skip] sentencepiece unavailable"); return
    from train_tokenizer import train_tokenizer
    from aeon.tokenizer import AeonTokenizer, PAD_ID, UNK_ID, BOS_ID, EOS_ID

    with tempfile.TemporaryDirectory() as d:
        corpus = os.path.join(d, "corpus.txt")
        _varied_corpus(corpus)
        model_path = train_tokenizer(corpus, os.path.join(d, "tok"),
                                     vocab_size=90, quiet=True)
        assert os.path.exists(model_path)
        tok = AeonTokenizer(model_path)

        # (1) requested vocab size
        assert tok.vocab_size == 90, tok.vocab_size
        # (2) fixed special-id layout
        assert (tok.pad_id, tok.unk_id, tok.bos_id, tok.eos_id) == (PAD_ID, UNK_ID, BOS_ID, EOS_ID)
        assert (PAD_ID, UNK_ID, BOS_ID, EOS_ID) == (0, 1, 2, 3)
        # (3) lossless round-trip on in-vocabulary text
        s = "Aeon holds the certificate margin."
        assert tok.decode(tok.encode(s)) == s, tok.decode(tok.encode(s))
        # add_bos/add_eos
        ids = tok.encode(s, add_bos=True, add_eos=True)
        assert ids[0] == BOS_ID and ids[-1] == EOS_ID
        # decode strips control ids (bos/eos/pad never surface as text)
        assert tok.decode(ids) == s


def test_data_reader_forms():
    """aeon/data.py reads .txt, .jsonl and directories into the same record stream."""
    import json
    from aeon.data import iter_text_records, corpus_files

    with tempfile.TemporaryDirectory() as d:
        txt = os.path.join(d, "a.txt")
        with open(txt, "w") as f:
            f.write("first line\nsecond line\n\n")           # blank line skipped
        jsonl = os.path.join(d, "b.jsonl")
        with open(jsonl, "w") as f:
            f.write(json.dumps({"text": "json record"}) + "\n")
            f.write(json.dumps({"text": ""}) + "\n")          # empty text skipped

        assert list(iter_text_records(txt)) == ["first line", "second line"]
        assert list(iter_text_records(jsonl)) == ["json record"]
        # directory: both files, sorted; 2 + 1 = 3 records
        recs = list(iter_text_records(d))
        assert sorted(recs) == ["first line", "json record", "second line"], recs
        assert len(corpus_files(d)) == 2


def _run_all():
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
