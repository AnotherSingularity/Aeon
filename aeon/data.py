"""
aeon/data.py — Aeon's corpus reader.

A tiny, dependency-free reader over the text forms a from-scratch corpus pipeline
naturally emits, shared by the tokenizer trainer and the training loop so there is
one definition of "what a corpus looks like":

  * a single .txt file   — one text record per line
  * a single .jsonl file — one JSON object per line, text taken from "text"
  * a directory          — every *.txt and *.jsonl under it, in sorted order

Nothing external, no downloads. When Dylan's synthetic-expansion pipeline fixes a
concrete on-disk format, this is the one place to extend (e.g. pre-tokenized
shards for a full 5–10B-token single-epoch run).
"""
from __future__ import annotations

import glob
import json
import os
from typing import Iterator, List


def corpus_files(path: str) -> List[str]:
    """Resolve `path` (file or directory) to the list of corpus files."""
    if os.path.isdir(path):
        files = sorted(glob.glob(os.path.join(path, "**", "*.txt"), recursive=True)
                       + glob.glob(os.path.join(path, "**", "*.jsonl"), recursive=True))
    elif os.path.exists(path):
        files = [path]
    else:
        raise FileNotFoundError(f"corpus path not found: {path!r}")
    if not files:
        raise FileNotFoundError(f"no .txt/.jsonl corpus files found under {path!r}")
    return files


def iter_text_records(path: str) -> Iterator[str]:
    """Yield one non-empty text record per line across all corpus files. .jsonl
    lines yield their "text" field; .txt lines yield the line verbatim."""
    for fp in corpus_files(path):
        is_jsonl = fp.endswith(".jsonl")
        with open(fp, "r", encoding="utf-8") as fh:
            for line in fh:
                if is_jsonl:
                    line = line.strip()
                    if not line:
                        continue
                    text = json.loads(line).get("text", "")
                else:
                    text = line.rstrip("\n")
                if text.strip():
                    yield text
