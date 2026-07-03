#!/usr/bin/env python3
"""
scripts/train_tokenizer.py — train Aeon's tokenizer from scratch.

Aeon trains its OWN tokenizer on Aeon's corpus — nothing adopted, nothing
downloaded. This trains a SentencePiece model (default 32k vocab) over a text
corpus and writes `<out>/<name>.model` + `<out>/<name>.vocab`, which
`aeon.tokenizer.AeonTokenizer` (and thus scripts/train.py, scripts/infer.py)
load directly. Version the produced `.model` alongside the weights.

CORPUS INPUT (accepts what a from-scratch corpus pipeline naturally emits):
  * a single .txt file (one document or line per line), or
  * a single .jsonl file with a "text" field per line, or
  * a directory — every *.txt and *.jsonl under it is used.

Special-id layout is fixed to match aeon/tokenizer.py:
    pad=0  unk=1  bos=2  eos=3   (ordinary pieces start at id 4)

Example:
    python scripts/train_tokenizer.py --corpus data/aeon_corpus \\
        --out tokenizer --name aeon --vocab-size 32000
"""
import argparse
import os
import tempfile

from aeon.data import corpus_files, iter_text_records
from aeon.tokenizer import PAD_ID, UNK_ID, BOS_ID, EOS_ID


def train_tokenizer(corpus, out, name="aeon", vocab_size=128000, model_type="unigram",
                    character_coverage=0.9999, input_sentence_size=0,
                    byte_fallback=True, quiet=False):
    """Train an Aeon SentencePiece tokenizer from `corpus`, writing
    `<out>/<name>.model` (+ `.vocab`). Returns the `.model` path. Importable so
    tests and pipelines can call it directly (main() is a thin CLI wrapper).

    Multilingual defaults (top-50 languages): 128k vocab, character_coverage
    0.9999, and byte_fallback so any code point (CJK/Indic/Arabic/…) that misses
    the piece vocabulary decomposes to UTF-8 bytes instead of <unk>."""
    import sentencepiece as spm  # deferred so importing this module needs no backend

    os.makedirs(out, exist_ok=True)
    files = corpus_files(corpus)

    # SentencePiece trains from a flat text file; normalise every input form into
    # one temp file of one-record-per-line so .jsonl and directories work too.
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        n = 0
        for text in iter_text_records(corpus):
            tmp.write(text.replace("\n", " ") + "\n")
            n += 1
        tmp_path = tmp.name
    if n == 0:
        raise ValueError("corpus produced 0 non-empty records")
    if not quiet:
        print(f"[tok] {len(files)} file(s) -> {n} records; training {model_type} "
              f"vocab={vocab_size} byte_fallback={byte_fallback}")

    prefix = os.path.join(out, name)
    spm.SentencePieceTrainer.train(
        input=tmp_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=bool(input_sentence_size),
        byte_fallback=byte_fallback,          # multilingual: no <unk>, bytes instead
        # fixed special-id layout — must match aeon/tokenizer.py
        pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
    )
    os.unlink(tmp_path)
    if not quiet:
        print(f"[tok] wrote {prefix}.model + {prefix}.vocab")
    return prefix + ".model"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="text file, .jsonl file, or a directory of them")
    ap.add_argument("--out", default="tokenizer", help="output directory for the .model/.vocab")
    ap.add_argument("--name", default="aeon", help="model prefix (produces <name>.model/.vocab)")
    ap.add_argument("--vocab-size", type=int, default=128000)
    ap.add_argument("--model-type", default="unigram", choices=["unigram", "bpe"])
    ap.add_argument("--character-coverage", type=float, default=0.9999)
    ap.add_argument("--input-sentence-size", type=int, default=0,
                    help="cap sentences sampled for training (0 = all)")
    ap.add_argument("--no-byte-fallback", dest="byte_fallback", action="store_false",
                    help="disable UTF-8 byte fallback (on by default for multilingual)")
    args = ap.parse_args()

    model_path = train_tokenizer(
        args.corpus, args.out, name=args.name, vocab_size=args.vocab_size,
        model_type=args.model_type, character_coverage=args.character_coverage,
        input_sentence_size=args.input_sentence_size, byte_fallback=args.byte_fallback)

    # sanity: round-trip through the Aeon wrapper
    from aeon.tokenizer import AeonTokenizer
    tok = AeonTokenizer(model_path)
    sample = "Aeon is its own architecture."
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    print(f"[tok] vocab_size={tok.vocab_size} | '{sample}' -> {ids} -> '{tok.decode(ids)}'")


if __name__ == "__main__":
    main()
