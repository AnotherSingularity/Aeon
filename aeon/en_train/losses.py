"""aeon.en_train.losses — L_G / L_C, effective-token accounting, sequence
buckets, mixture sampler.

Implements §5, §6, §7, §8, §11 without introducing any transformer-only
auxiliary loss. Every loss consumes ``HybridModel.forward`` output
logits directly. Nothing is inserted between logits and the loss.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# §5 general-English loss
# ---------------------------------------------------------------------------
def masked_next_token_loss(logits: torch.Tensor,
                                targets: torch.Tensor,
                                mask: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """Standard shifted cross-entropy with a per-position mask.

    logits  : (B, L, V)
    targets : (B, L)   — already shifted so targets[:, t] is the target
                          for prediction at logit position t.
    mask    : (B, L)   — 1 = valid supervised position, 0 = padding /
                          not supervised.

    Returns:
      (mean_loss_over_valid_positions_only, valid_token_count)
    """
    B, L, V = logits.shape
    logp = F.log_softmax(logits.float(), dim=-1)
    # Gather log p at target positions
    tgt = targets.long().clamp(min=0, max=V - 1).unsqueeze(-1)  # (B,L,1)
    logp_at_tgt = logp.gather(-1, tgt).squeeze(-1)               # (B,L)
    m = mask.float()
    valid = m.sum()
    if float(valid.item()) == 0.0:
        return logp_at_tgt.sum() * 0.0, 0
    loss = -(logp_at_tgt * m).sum() / valid
    return loss, int(valid.item())


def general_english_loss(model, input_ids: torch.Tensor,
                              attention_mask: Optional[torch.Tensor] = None
                              ) -> Tuple[torch.Tensor, int]:
    """§5: L_G(θ). Predict token t+1 from x_{1..t} for every non-padding t."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]              # positions 0..L-2 predict 1..L-1
    targets = input_ids[:, 1:]                  # (B, L-1)
    mask = attention_mask[:, 1:].float()        # (B, L-1)
    return masked_next_token_loss(logits, targets, mask)


# ---------------------------------------------------------------------------
# §6 response-masked conversational loss
# ---------------------------------------------------------------------------
def conversational_loss(model, input_ids: torch.Tensor,
                             response_mask: torch.Tensor,
                             attention_mask: Optional[torch.Tensor] = None
                             ) -> Tuple[torch.Tensor, int]:
    """§6: L_C(θ). ``response_mask[b, t] = 1`` means the loss is
    computed at prediction position t (i.e. target token = input_ids[b, t+1]
    is part of the human assistant response). User turns remain context."""
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    # response_mask (B, L). The supervised position for target `t+1`
    # is prediction position `t`. So we take response_mask[:, 1:] and
    # AND it with attention_mask[:, 1:].
    m = (response_mask[:, 1:] * attention_mask[:, 1:]).float()
    return masked_next_token_loss(logits, targets, m)


# ---------------------------------------------------------------------------
# §11 effective-token accounting
# ---------------------------------------------------------------------------
@dataclass
class EffectiveTokenCounter:
    tokens_this_update: int = 0
    updates: int = 0
    total_tokens: int = 0

    def add_microbatch(self, valid_token_count: int) -> None:
        self.tokens_this_update += int(valid_token_count)

    def commit_update(self) -> int:
        n = self.tokens_this_update
        self.tokens_this_update = 0
        self.updates += 1
        self.total_tokens += n
        return n


# ---------------------------------------------------------------------------
# §8 sequence-bucket construction
# ---------------------------------------------------------------------------
BUCKET_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
BUCKET_WEIGHTS = (0.25, 0.25, 0.25, 0.25)


def pick_sequence_length(l_native: int, rng: random.Random) -> int:
    """Choose a sequence length within ±10% of a bucket target,
    per §8."""
    bucket = rng.choices(BUCKET_FRACTIONS, weights=BUCKET_WEIGHTS, k=1)[0]
    target = max(1, int(round(bucket * l_native)))
    jitter = max(1, int(round(0.10 * target)))
    return max(1, min(l_native, target + rng.randint(-jitter, jitter)))


# ---------------------------------------------------------------------------
# §7 stage mixture sampler
# ---------------------------------------------------------------------------
class StageMixture:
    def __init__(self, weights: Dict[str, float], seed: int = 20260803):
        s = sum(weights.values())
        assert abs(s - 1.0) < 1e-6, f"mixture weights must sum to 1, got {s}"
        self.keys = tuple(weights.keys())
        self.probs = tuple(weights[k] for k in self.keys)
        self.rng = random.Random(seed)

    def pick(self) -> str:
        return self.rng.choices(self.keys, weights=self.probs, k=1)[0]


# ---------------------------------------------------------------------------
# Chat serialization for L_C (§6 — marks speaker boundaries only, never
# inserts answer content).
# ---------------------------------------------------------------------------
# We use text-only role markers because the tokenizer was not trained
# with dedicated chat tokens. Concrete choice:
#   turn boundary        : "\n\n"
#   user role prefix     : "user: "
#   assistant role prefix: "assistant: "
# The response_mask is computed over the character positions belonging
# to the assistant's response (and its trailing newline), then converted
# to per-token by re-encoding.
USER_PREFIX = "user: "
ASSIST_PREFIX = "assistant: "
TURN_SEP = "\n\n"


def render_conversation_for_training(turns: Sequence[Tuple[str, str]]
                                              ) -> Tuple[str, List[Tuple[int, int]]]:
    """Return (text, list_of_(assistant_span_start, assistant_span_end))
    in CHARACTER units. Turns is a sequence of (role, content) with
    role in {'user','assistant'}."""
    parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for i, (role, content) in enumerate(turns):
        prefix = USER_PREFIX if role == "user" else ASSIST_PREFIX
        segment = prefix + content
        if i > 0:
            parts.append(TURN_SEP); cursor += len(TURN_SEP)
        assistant_start = cursor + len(prefix) if role == "assistant" else None
        parts.append(segment); cursor += len(segment)
        if role == "assistant":
            spans.append((assistant_start, cursor))
    return "".join(parts), spans


def build_response_mask(tokenizer, text: str,
                              assistant_char_spans: Sequence[Tuple[int, int]]
                              ) -> Tuple[List[int], List[int]]:
    """Encode `text` and produce a per-token response_mask that is 1
    wherever the token's characters fall inside any assistant span.

    Returns (token_ids, response_mask).

    Because SentencePiece tokens can straddle character boundaries, a
    token is marked supervised only if it lies FULLY inside a span.
    """
    ids = tokenizer.encode(text, add_bos=False, add_eos=False)
    # SentencePiece doesn't expose per-token character offsets directly.
    # Re-decode piece-by-piece to compute cumulative character offsets.
    sp = tokenizer._sp
    pieces = [sp.id_to_piece(i) for i in ids]
    # sentencepiece marks spaces with ▁ ('▁'); decoding one piece
    # at a time via decode_ids preserves the runtime's semantics.
    offsets: List[Tuple[int, int]] = []
    cursor = 0
    for i, _piece in enumerate(pieces):
        # Decode the cumulative prefix and take the tail
        # (matches the streaming renderer's semantics).
        piece_text = sp.decode(ids[:i + 1])[cursor:]
        end = cursor + len(piece_text)
        offsets.append((cursor, end))
        cursor = end

    def _fully_inside(o: Tuple[int, int]) -> bool:
        for a, b in assistant_char_spans:
            if o[0] >= a and o[1] <= b:
                return True
        return False

    mask = [1 if _fully_inside(o) else 0 for o in offsets]
    return list(ids), mask
