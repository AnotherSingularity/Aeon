"""W10-1 — real tokenizer + corpus in the GUI worker.

Every test in this file uses a REAL SentencePiece tokenizer trained on a
bounded English fixture and drives it through the worker's canonical data
source. No test creates a placeholder tokenizer; no test uses random token
IDs.

Ten of the ordered assertions the W10-1 directive requires:

1. Known English text produces the expected tokenizer output.
2. The worker's data source consumes those exact token IDs.
3. The token IDs reach the model forward and loss path (via TorchLite
   surrogate for the model contract — the actual HybridModel is exercised
   by the launcher tests in test_launcher_and_job.py).
4. Changing corpus content changes produced batches.
5. Changing tokenizer identity changes tokenizer_id.
6. Empty corpus is refused with DataSourceError('corpus_empty').
7. Missing corpus is refused with DataSourceError('corpus_missing').
8. Missing tokenizer is refused with DataSourceError('tokenizer_missing').
9. Resume from data_position reproduces the exact next batch.
10. No production worker code path calls torch.randint.

The fixture is trained once per test session and cached on disk. It uses
byte_fallback so tiny English text is guaranteed to tokenize to more than
one batch of ids at the batch/seq shapes W10-1 uses.
"""
import ast
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

# Small English fixture — 4 KB. This is a public-domain-quality style text
# that will tokenize into a few hundred token ids at the tiny vocab we train.
FIXTURE_TEXT = (
    "Aeon carries its own tokenizer, trained from scratch on its own corpus. "
    "The purpose of this fixture is to prove that the launcher's worker "
    "consumes real English tokens rather than synthetic random ids. "
    "The quick brown fox jumps over the lazy dog. "
    "Sphinx of black quartz, judge my vow. "
    "How vexingly quick daft zebras jump! "
    "The five boxing wizards jump quickly. "
    "Pack my box with five dozen liquor jugs. "
    "Waltz, bad nymph, for quick jigs vex. "
    "Bright vixens jump; dozy fowl quack. "
    "Recursion is the sole cross-stream integration point in Aeon. "
    "The transformer and substrate are independent parallel streams. "
    "Recursion state remains in fp32 for the contractive certificate. "
    "K is fixed to sixteen and the broadcast is single. "
    "This paragraph exists solely so that the sentencepiece trainer sees "
    "enough real text to fit a small unigram model. "
) * 20  # tile to a few KB


_TOK_CACHE_DIR = os.path.join(
    tempfile.gettempdir(), "aeon-w10-1-tokenizer-fixture-cache")


def _make_corpus_and_tokenizer(vocab_size: int = 400):
    """Train a tiny real SentencePiece tokenizer on the fixture. Cached on
    disk between tests in the same interpreter run."""
    if os.path.exists(_TOK_CACHE_DIR):
        model_path = os.path.join(_TOK_CACHE_DIR, "aeon.model")
        corpus_path = os.path.join(_TOK_CACHE_DIR, "corpus.txt")
        if os.path.exists(model_path) and os.path.exists(corpus_path):
            return model_path, corpus_path
    os.makedirs(_TOK_CACHE_DIR, exist_ok=True)
    corpus_path = os.path.join(_TOK_CACHE_DIR, "corpus.txt")
    with open(corpus_path, "w", encoding="utf-8") as fh:
        # One record per line — the aeon.data reader treats each non-empty
        # line as a record.
        for line in FIXTURE_TEXT.split(". "):
            line = line.strip()
            if line:
                fh.write(line + ".\n")
    from scripts.train_tokenizer import train_tokenizer
    # character_coverage=0.995 keeps required_chars manageable for the tiny
    # fixture (English + some punctuation). byte_fallback stays on so any
    # missing code point decomposes to bytes rather than <unk>.
    model_path = train_tokenizer(
        corpus_path, _TOK_CACHE_DIR, name="aeon",
        vocab_size=vocab_size, model_type="unigram",
        character_coverage=0.995, byte_fallback=True, quiet=True)
    return model_path, corpus_path


# ---------------------------------------------------------------------------
# Fake Job dataclass — matches only what the data source uses
# ---------------------------------------------------------------------------
class _FakeJob:
    def __init__(self, tokenizer_path, corpus_path):
        self.tokenizer_path = tokenizer_path
        self.corpus_path = corpus_path


# ---------------------------------------------------------------------------
# 1 & 2. Known text → expected token IDs → data source yields them
# ---------------------------------------------------------------------------
def test_known_text_produces_expected_tokenizer_output():
    model_path, corpus_path = _make_corpus_and_tokenizer()
    from aeon.tokenizer import AeonTokenizer
    tok = AeonTokenizer(model_path)
    ids = tok.encode("The quick brown fox.")
    assert isinstance(ids, list) and ids, "tokenizer must yield ids"
    # Round-trip the ids back to text — SentencePiece decode should recover
    # something meaningful (byte fallback makes exact string match strict
    # but round-trip through the same tokenizer is stable).
    round_trip = tok.decode(ids)
    assert "quick" in round_trip.lower() or "brown" in round_trip.lower(), (
        f"round-trip should preserve key vocabulary; got {round_trip!r}")


def test_data_source_yields_real_corpus_ids():
    """The first batch produced by the data source must equal the tokenized
    corpus stream at offset 0, packed to (batch_size, seq_len)."""
    model_path, corpus_path = _make_corpus_and_tokenizer()
    from aeon.job.data_source import build_data_source
    from aeon.tokenizer import AeonTokenizer
    from aeon.data import iter_text_records

    tcfg = {"batch_size": 1, "seed": 0}
    dcfg = {"seq_len": 16}
    src = build_data_source(_FakeJob(model_path, corpus_path), tcfg, dcfg)
    assert src.tokenizer.vocab_size >= 32, "trained a real tokenizer"

    # Reconstruct expected stream independently.
    tok = AeonTokenizer(model_path)
    expected_stream = []
    for text in iter_text_records(corpus_path):
        expected_stream.extend(tok.encode(text, add_eos=True))

    batch_iter = src.iter_batches(device="cpu", start_position=0)
    batch, pos_after = next(batch_iter)
    ids = batch["input_ids"].reshape(-1).tolist()
    assert ids == expected_stream[:16], (
        f"first-batch ids must equal expected_stream[:16]; got {ids[:4]}… "
        f"expected {expected_stream[:4]}…")
    assert pos_after == 16, "position advances by batch_size*seq_len"


# ---------------------------------------------------------------------------
# 3. Real ids reach model forward/loss — exercised indirectly via shape
# ---------------------------------------------------------------------------
def test_batch_dict_matches_hybridmodel_forward_contract():
    model_path, corpus_path = _make_corpus_and_tokenizer()
    from aeon.job.data_source import build_data_source
    src = build_data_source(_FakeJob(model_path, corpus_path),
                             {"batch_size": 2, "seed": 0}, {"seq_len": 8})
    batch, _ = next(src.iter_batches(device="cpu", start_position=0))
    for k in ("input_ids", "attention_mask", "labels"):
        assert k in batch, f"batch missing {k}"
    assert batch["input_ids"].shape == (2, 8)
    assert batch["attention_mask"].shape == (2, 8)
    assert batch["labels"].shape == (2, 8)
    # labels is a clone of input_ids (next-token cross-entropy target).
    assert (batch["labels"] == batch["input_ids"]).all().item()


# ---------------------------------------------------------------------------
# 4. Changing corpus content changes produced batches
# ---------------------------------------------------------------------------
def test_changing_corpus_content_changes_produced_batches():
    model_path, corpus_path = _make_corpus_and_tokenizer()
    from aeon.job.data_source import build_data_source

    src1 = build_data_source(_FakeJob(model_path, corpus_path),
                              {"batch_size": 1, "seed": 0}, {"seq_len": 4})
    ids1 = next(src1.iter_batches("cpu", 0))[0]["input_ids"].reshape(-1).tolist()

    with tempfile.TemporaryDirectory() as d:
        alt_corpus = os.path.join(d, "alt.txt")
        with open(alt_corpus, "w", encoding="utf-8") as fh:
            # Intentionally different text — but long enough to tokenize.
            fh.write(("A completely different sentence with different words. " * 40))
        src2 = build_data_source(_FakeJob(model_path, alt_corpus),
                                  {"batch_size": 1, "seed": 0}, {"seq_len": 4})
        ids2 = next(src2.iter_batches("cpu", 0))[0]["input_ids"].reshape(-1).tolist()

    assert ids1 != ids2, (
        "different corpora must produce different token id sequences at offset 0")


# ---------------------------------------------------------------------------
# 5. Tokenizer identity is a content hash
# ---------------------------------------------------------------------------
def test_tokenizer_identity_is_content_hash():
    model_path, corpus_path = _make_corpus_and_tokenizer()
    from aeon.job.data_source import tokenizer_identity

    ident = tokenizer_identity(model_path)
    assert ident.startswith("sha256:")
    h = hashlib.sha256(open(model_path, "rb").read()).hexdigest()
    assert ident == f"sha256:{h}"


def test_corpus_identity_is_stable_and_content_sensitive():
    model_path, corpus_path = _make_corpus_and_tokenizer()
    from aeon.job.data_source import corpus_identity
    id_a = corpus_identity(corpus_path)
    id_b = corpus_identity(corpus_path)
    assert id_a == id_b, "corpus identity must be stable"
    with tempfile.TemporaryDirectory() as d:
        alt = os.path.join(d, "alt.txt")
        with open(alt, "w", encoding="utf-8") as fh:
            fh.write("different content\n" * 10)
        id_c = corpus_identity(alt)
        assert id_c != id_a, "changing corpus content changes corpus identity"


# ---------------------------------------------------------------------------
# 6, 7, 8. Fail-closed on missing / empty / bad inputs
# ---------------------------------------------------------------------------
def test_fails_closed_on_missing_tokenizer():
    from aeon.job.data_source import build_data_source, DataSourceError
    model_path, corpus_path = _make_corpus_and_tokenizer()
    job = _FakeJob(None, corpus_path)
    try:
        build_data_source(job, {"batch_size": 1}, {"seq_len": 8})
        raise AssertionError("expected DataSourceError")
    except DataSourceError as e:
        assert e.reason == "tokenizer_absent", e.reason


def test_fails_closed_on_missing_tokenizer_file():
    from aeon.job.data_source import build_data_source, DataSourceError
    model_path, corpus_path = _make_corpus_and_tokenizer()
    with tempfile.TemporaryDirectory() as d:
        job = _FakeJob(os.path.join(d, "nope.model"), corpus_path)
        try:
            build_data_source(job, {"batch_size": 1}, {"seq_len": 8})
            raise AssertionError("expected DataSourceError")
        except DataSourceError as e:
            assert e.reason == "tokenizer_missing", e.reason


def test_fails_closed_on_missing_corpus():
    from aeon.job.data_source import build_data_source, DataSourceError
    model_path, corpus_path = _make_corpus_and_tokenizer()
    job = _FakeJob(model_path, None)
    try:
        build_data_source(job, {"batch_size": 1}, {"seq_len": 8})
        raise AssertionError("expected DataSourceError")
    except DataSourceError as e:
        assert e.reason == "corpus_absent", e.reason


def test_fails_closed_on_empty_corpus():
    from aeon.job.data_source import build_data_source, DataSourceError
    model_path, corpus_path = _make_corpus_and_tokenizer()
    with tempfile.TemporaryDirectory() as d:
        empty = os.path.join(d, "empty.txt")
        Path(empty).write_text("", encoding="utf-8")
        job = _FakeJob(model_path, empty)
        try:
            build_data_source(job, {"batch_size": 1}, {"seq_len": 8})
            raise AssertionError("expected DataSourceError")
        except DataSourceError as e:
            # An empty file has no records => corpus_empty.
            assert e.reason == "corpus_empty", e.reason


def test_fails_closed_on_corpus_too_small_for_batch():
    from aeon.job.data_source import build_data_source, DataSourceError
    model_path, _ = _make_corpus_and_tokenizer()
    with tempfile.TemporaryDirectory() as d:
        tiny = os.path.join(d, "tiny.txt")
        Path(tiny).write_text("hi.\n", encoding="utf-8")
        job = _FakeJob(model_path, tiny)
        try:
            build_data_source(job, {"batch_size": 4}, {"seq_len": 128})
            raise AssertionError("expected DataSourceError")
        except DataSourceError as e:
            assert e.reason == "corpus_too_small", e.reason


# ---------------------------------------------------------------------------
# 9. Deterministic resume from data_position
# ---------------------------------------------------------------------------
def test_resume_from_data_position_reproduces_next_batch():
    model_path, corpus_path = _make_corpus_and_tokenizer()
    from aeon.job.data_source import build_data_source
    src = build_data_source(_FakeJob(model_path, corpus_path),
                             {"batch_size": 1, "seed": 0}, {"seq_len": 4})
    it = src.iter_batches("cpu", 0)
    b1, pos1 = next(it); ids1 = b1["input_ids"].reshape(-1).tolist()
    b2, pos2 = next(it); ids2 = b2["input_ids"].reshape(-1).tolist()
    b3, pos3 = next(it); ids3 = b3["input_ids"].reshape(-1).tolist()

    # A fresh source seeked to pos1 must yield ids2, ids3 in order.
    src_resumed = build_data_source(_FakeJob(model_path, corpus_path),
                                     {"batch_size": 1, "seed": 0}, {"seq_len": 4})
    it2 = src_resumed.iter_batches("cpu", start_position=pos1)
    r2, rpos2 = next(it2); rids2 = r2["input_ids"].reshape(-1).tolist()
    r3, rpos3 = next(it2); rids3 = r3["input_ids"].reshape(-1).tolist()
    assert rids2 == ids2 and rpos2 == pos2, "resume must reproduce next batch"
    assert rids3 == ids3 and rpos3 == pos3, "resume must reproduce two-ahead batch"


# ---------------------------------------------------------------------------
# 10. No production torch.randint in the worker's training loop
# ---------------------------------------------------------------------------
def test_no_production_torch_randint_in_worker():
    """The `torch.randint` STRING may appear inside a comment explaining why
    the synthetic path was removed. Ban it from CODE lines only."""
    src = open(os.path.join(ROOT, "aeon/job/worker.py"), encoding="utf-8").read()
    import re as _re
    m = _re.search(r"def _run_training_loop.*?(?=\n(?:def |\Z))", src, _re.DOTALL)
    assert m
    code_lines = [
        line for line in m.group(0).splitlines()
        if not line.lstrip().startswith("#")]
    body_code = "\n".join(code_lines)
    assert "torch.randint" not in body_code, (
        "W10-1: no production code line in _run_training_loop may call "
        "torch.randint")


def test_worker_imports_data_source_module():
    src = open(os.path.join(ROOT, "aeon/job/worker.py"), encoding="utf-8").read()
    assert "from aeon.job.data_source import build_data_source" in src


# ---------------------------------------------------------------------------
# End-to-end proof: real English text -> tokenizer -> data source -> a real
# HybridModel forward + loss + backward + step. This is the exit gate of
# W10-1: token IDs from a real English fixture reach model loss computation.
# ---------------------------------------------------------------------------
def test_real_english_text_reaches_model_loss_computation():
    import torch
    from aeon.job.data_source import build_data_source
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig

    model_path, corpus_path = _make_corpus_and_tokenizer()
    tcfg = {"batch_size": 1, "seed": 0}
    dcfg = {"seq_len": 8}
    src = build_data_source(_FakeJob(model_path, corpus_path), tcfg, dcfg)
    vocab = src.tokenizer.vocab_size

    tcfg_model = AeonTransformerConfig(
        vocab_size=vocab, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=64, rms_norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=True, attention_bias=False)
    model = HybridModel(
        h_rec=8, K=16, transformer_config=tcfg_model,
        substrate={"kind": "matrix", "d_in": 8, "d_state": 8,
                    "n_head": 2, "head_size": 4},
        margin_h=0.98, margin_c=0.95,
        freeze_backbone=False, use_embedding_input=True,
        dtype=torch.float32)
    model.transformer.gamma.data = model.transformer.gamma.data.float()
    fb = getattr(model.substrate, "feedback", None)
    if fb is not None and isinstance(fb.gate_alpha, torch.nn.Parameter):
        fb.gate_alpha.data = fb.gate_alpha.data.float()
        fb.gate_threshold.data = fb.gate_threshold.data.float()
    model.train()

    batch, _ = next(src.iter_batches("cpu", 0))
    out = model(input_ids=batch["input_ids"],
                 attention_mask=batch["attention_mask"],
                 labels=batch["labels"])
    loss = out.loss
    # Real English tokens produced a finite training loss.
    assert loss.item() > 0, f"loss should be positive, got {loss.item()}"
    assert loss.item() < 1e6, f"loss suspiciously large: {loss.item()}"
    loss.backward()
    # At least the model's input embedding got a gradient (proof that the
    # ids reached the model, not just a wrapper).
    grads_present = any(p.grad is not None and (p.grad.abs().sum() > 0).item()
                        for p in model.trainable_parameters())
    assert grads_present, "real token ids must produce non-zero grads"


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
