"""scripts/en_train_validate_intake.py — verify a corpus intake dir.

Usage:
    python scripts/en_train_validate_intake.py \\
        --intake research-data/incoming/<CORPUS_ID>

Runs the §3 layout + provenance validator + §4 tokenizer r_UNK gate
on the sources. Reports go to stdout. Does not train."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aeon.en_train.data import validate_intake_layout, run_tokenizer_check, Document
from aeon.tokenizer import AeonTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--intake", required=True)
    ap.add_argument("--tokenizer",
                     default="research-data/AEON-LBC-1/tokenizer/aeon-lbc1.model")
    args = ap.parse_args()

    layout = validate_intake_layout(Path(args.intake))
    print(json.dumps({"layout": layout}, indent=2, sort_keys=True))

    tok = AeonTokenizer(args.tokenizer)
    docs = []
    for i, src in enumerate(layout["sources"]):
        with open(src["path"], "rb") as f:
            raw = f.read()
        text = raw.decode("utf-8", errors="strict")
        docs.append(Document(doc_id=src["source_id"], text=text,
                                  source_id=src["source_id"],
                                  author_or_institution=src["author_or_institution"],
                                  est_token_count=max(1, len(text) // 4)))
    r = run_tokenizer_check(tok, docs)
    print(json.dumps({
        "tokenizer_check": {
            "total_tokens": r.total_tokens,
            "unk_tokens": r.unk_tokens,
            "r_unk": r.r_unk,
            "max_id_seen": r.max_id_seen,
            "passed_r_unk_gate": r.passed,
        }
    }, indent=2, sort_keys=True))
    return 0 if r.passed else 4


if __name__ == "__main__":
    sys.exit(main())
