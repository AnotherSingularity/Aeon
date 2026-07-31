"""aeon/job/data_source.py — real fail-closed corpus batch iterator (W10-1).

Owns the ONLY production data path the GUI worker is allowed to use. Reuses
the canonical implementation from ``scripts/train.py::iter_corpus_batches``
via ``aeon.data.iter_text_records`` + ``aeon.tokenizer.AeonTokenizer``, so
the launcher path and the CLI training path emit byte-identical batches for
the same (tokenizer, corpus, seed, batch_size, seq_len, start_position).

Fail-closed policy (§W10-1):

* Absent, missing, unreadable, or wrong-shaped tokenizer -> DataSourceError.
* Absent, missing, empty, or unreadable corpus -> DataSourceError.
* Zero corpus records -> DataSourceError.
* Corpus that tokenizes to fewer than one batch of tokens -> DataSourceError.
* Optional corpus_manifest_path that fails F2 provenance validation
  -> DataSourceError.

No synthetic fallback. No ``torch.randint`` production path. Synthetic
sources exist only under explicit test-fixture opt-in (see
``tests/test_w10_1_real_corpus.py``), never selectable from the GUI or from
worker configuration.

Determinism guarantee: given a fixed (tokenizer file bytes, corpus file
bytes, batch_size, seq_len), the sequence of yielded token IDs is
deterministic. ``start_position`` seeks into that stream and the resumed
worker produces the exact same next batch as an uninterrupted run.
"""
from __future__ import annotations

import hashlib
import os
from typing import Iterator, List, Optional, Tuple


class DataSourceError(RuntimeError):
    """Raised when a required tokenizer or corpus cannot be used.

    Carries a short, structured ``reason`` code so the worker can write it
    into ``result.json``/``status.json`` for the launcher to surface.
    """

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_identity(tokenizer_path: str) -> str:
    """Stable identity of a tokenizer artifact — sha256 of the .model bytes.

    The Aeon tokenizer is a SentencePiece ``.model`` file; hashing its bytes
    binds an exact tokenizer to an exact checkpoint. If the tokenizer changes
    even by one byte, the identity changes, so downstream checkpoint-vs-
    tokenizer compatibility checks catch mismatch."""
    return f"sha256:{_sha256_file(tokenizer_path)}"


def corpus_identity(corpus_path: str) -> str:
    """Stable identity of a corpus tree — sha256 over the sorted list of
    per-file sha256:filename lines. Order-invariant, but sensitive to file
    contents and to added/removed files."""
    from aeon.data import corpus_files

    files = corpus_files(corpus_path)  # raises FileNotFoundError if empty
    parts = []
    for fp in files:
        parts.append(f"{_sha256_file(fp)}  {os.path.basename(fp)}")
    aggregate = hashlib.sha256("\n".join(sorted(parts)).encode("utf-8"))
    return f"sha256:{aggregate.hexdigest()}"


class TokenizedCorpusBatchSource:
    """Deterministic real-corpus batch iterator.

    Instances are constructed by ``build_data_source()``. Callers must not
    reach past the factory to synthesise a source with a fake tokenizer.
    """

    def __init__(
        self,
        tokenizer,
        corpus_path: str,
        batch_size: int,
        seq_len: int,
        *,
        tokenizer_id: str,
        corpus_id: str,
        partition: str = "train",
    ):
        from aeon.data import iter_text_records

        self.batch_size = int(batch_size)
        self.seq_len = int(seq_len)
        self.tokenizer_id = tokenizer_id
        self.corpus_id = corpus_id
        self.partition = partition
        self.tokenizer = tokenizer

        # Tokenize the entire corpus once. This matches the canonical
        # scripts/train.py::iter_corpus_batches implementation — appropriate
        # for the prototype/sanity subset; a full corpus wants pre-tokenized
        # shards read lazily and is out of scope for W10-1.
        stream: List[int] = []
        n_records = 0
        for text in iter_text_records(corpus_path):
            stream.extend(tokenizer.encode(text, add_eos=True))
            n_records += 1
        if n_records == 0:
            raise DataSourceError(
                "corpus_empty",
                f"corpus at {corpus_path!r} produced 0 records")
        span = self.batch_size * self.seq_len
        if len(stream) < span + 1:
            raise DataSourceError(
                "corpus_too_small",
                f"corpus tokenized to {len(stream)} tokens < one batch ({span}+1)")
        self._stream = stream
        self.total_tokens = len(stream)
        self.records = n_records

    def iter_batches(
        self,
        device: str = "cpu",
        start_position: int = 0,
    ) -> Iterator[Tuple[dict, int]]:
        """Yield ``(batch_dict, position_after)`` pairs.

        ``batch_dict`` matches the shape scripts/train.py's HybridModel
        expects: ``input_ids``, ``attention_mask``, ``labels``. All three
        are shape ``(batch_size, seq_len)``. ``labels`` is a clone of
        ``input_ids`` so cross-entropy learns next-token prediction.

        ``position_after`` is the absolute token offset immediately after
        the yielded batch. A worker that saves ``position_after`` into
        checkpoint metadata and restores it via ``start_position=metadata
        ["data_position"]`` produces the exact same next batch that an
        uninterrupted run would have produced.
        """
        import torch  # deferred (worker-only heavy import)

        span = self.batch_size * self.seq_len
        pos = int(start_position)
        if pos < 0 or pos >= self.total_tokens - 1:
            raise DataSourceError(
                "start_position_out_of_range",
                f"pos={pos} not in [0, {self.total_tokens - 1})")

        while pos + span <= self.total_tokens:
            block = self._stream[pos : pos + span]
            ids = torch.tensor(block, dtype=torch.long, device=device).view(
                self.batch_size, self.seq_len)
            pos += span
            yield (
                {
                    "input_ids": ids,
                    "attention_mask": torch.ones_like(ids),
                    "labels": ids.clone(),
                },
                pos,
            )


def _validate_corpus_manifest(manifest_path: Optional[str]) -> None:
    """If a manifest path is provided, load and F2-validate it. Any error
    raises DataSourceError with reason='corpus_provenance_invalid'."""
    if not manifest_path:
        return
    if not os.path.exists(manifest_path):
        raise DataSourceError(
            "corpus_manifest_missing",
            f"corpus_manifest_path {manifest_path!r} not found")
    import json

    from aeon.corpus_manifest import validate_manifest

    try:
        manifest = json.load(open(manifest_path, encoding="utf-8"))
    except Exception as e:
        raise DataSourceError(
            "corpus_manifest_unreadable", f"{manifest_path!r}: {e}") from e
    errs = validate_manifest(manifest)
    if errs:
        raise DataSourceError(
            "corpus_provenance_invalid",
            "; ".join(errs[:5]) + ("; …" if len(errs) > 5 else ""))


def build_data_source(
    job,
    tcfg: dict,
    dcfg: dict,
    *,
    partition: str = "train",
) -> TokenizedCorpusBatchSource:
    """Fail-closed factory: assemble the ONE canonical batch source or raise.

    Rejects every path the audit flagged as production-reachable-synthetic:

    * job.tokenizer_path missing or empty
    * tokenizer file not on disk
    * job.corpus_path missing or empty
    * corpus file not on disk
    * corpus produces zero records
    * corpus tokenizes to fewer tokens than one batch
    * optional dcfg["corpus_manifest_path"] fails F2 provenance validation

    The returned source is the only object the worker's training loop is
    permitted to iterate over. No branch in the worker may substitute a
    synthetic source.
    """
    from aeon.tokenizer import AeonTokenizer

    tokenizer_path = getattr(job, "tokenizer_path", None)
    corpus_path = getattr(job, "corpus_path", None)

    if not tokenizer_path:
        raise DataSourceError(
            "tokenizer_absent",
            "job.tokenizer_path is required — W10-1 disallows synthetic training")
    if not os.path.exists(tokenizer_path):
        raise DataSourceError("tokenizer_missing", f"{tokenizer_path!r} not found")
    if not corpus_path:
        raise DataSourceError(
            "corpus_absent",
            "job.corpus_path is required — W10-1 disallows synthetic training")
    if not os.path.exists(corpus_path):
        raise DataSourceError("corpus_missing", f"{corpus_path!r} not found")

    _validate_corpus_manifest(dcfg.get("corpus_manifest_path"))

    try:
        tok = AeonTokenizer(tokenizer_path)
    except FileNotFoundError as e:
        raise DataSourceError("tokenizer_missing", str(e)) from e
    except Exception as e:
        raise DataSourceError("tokenizer_unreadable", repr(e)) from e

    batch_size = int(tcfg.get("batch_size", 1))
    seq_len = int(dcfg.get("seq_len", 512))
    if batch_size <= 0 or seq_len <= 0:
        raise DataSourceError(
            "batch_shape_invalid",
            f"batch_size={batch_size} seq_len={seq_len} must both be positive")

    tok_id = tokenizer_identity(tokenizer_path)
    corp_id = corpus_identity(corpus_path)

    return TokenizedCorpusBatchSource(
        tokenizer=tok,
        corpus_path=corpus_path,
        batch_size=batch_size,
        seq_len=seq_len,
        tokenizer_id=tok_id,
        corpus_id=corp_id,
        partition=partition,
    )
