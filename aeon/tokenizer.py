"""
aeon/tokenizer.py — Aeon's own tokenizer, trained from scratch on Aeon's corpus.

Aeon carries its from-scratch commitment down to the tokenizer: nothing adopted,
nothing repackaged from another model. This module is a thin, framework-free
wrapper over a SentencePiece model that Aeon TRAINS ITSELF (see
`scripts/train_tokenizer.py`) — it downloads nothing and depends on no external
tokenizer artifact. The trained `.model` is versioned alongside the weights.

Fixed special-id layout (stable across retrains, so checkpoints stay compatible):
    pad = 0    unk = 1    bos = 2    eos = 3
Ordinary pieces occupy ids >= 4.
"""
from __future__ import annotations

import os
from typing import List, Sequence

# Special-token ids are fixed by contract (see module docstring). The tokenizer
# trainer is configured to honour exactly this layout.
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class AeonTokenizer:
    """Loads an Aeon-trained SentencePiece model and exposes encode/decode.

    Import of `sentencepiece` is deferred to construction so that importing
    `aeon` (and its framework-free contracts) never requires the tokenizer
    backend to be installed.
    """

    def __init__(self, model_path: str):
        import sentencepiece as spm  # deferred; only needed to actually tokenize

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Aeon tokenizer model not found: {model_path!r}. Train one with "
                f"scripts/train_tokenizer.py (Aeon trains its own; nothing is downloaded)."
            )
        self.model_path = model_path
        self._sp = spm.SentencePieceProcessor(model_file=model_path)
        self.pad_id = PAD_ID
        self.unk_id = UNK_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID

    @property
    def vocab_size(self) -> int:
        return self._sp.vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = self._sp.encode(text, out_type=int)
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        # Drop control ids so they never surface as literal text.
        ids = [int(i) for i in ids if int(i) not in (self.pad_id, self.bos_id, self.eos_id)]
        return self._sp.decode(ids)

    def __len__(self) -> int:
        return self.vocab_size
