"""
aeon/transformer.py — the transformer side (Qwen2-family backbone).

⚠️ UNRUN. Written to spec for the Stage-1 hybrid; no torch/transformers and no
HuggingFace access in the authoring environment. Backbone load + forward are
verified on Vast (per the agreed plan).

WHAT THIS IS
------------
A thin wrapper around the `transformers` library Qwen2 implementation (we use
the library here for speed — this is NOT an Aeon-original rewrite). The backbone
is one signal source among several (not a privileged "reasoner"). Default
backbone: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (Stage-1 parity).

  * Backbone is FROZEN by default; only the read/write surfaces train.
  * READ port  : residual-stream hidden states → `read_proj` → manifold (H_rec).
  * WRITE port : γ-gated injection of a manifold-width signal back into the
                 hidden state, γ initialised at 0 ⇒ at init the model is
                 byte-identical to plain R1 (warm start).

This module exposes primitives (embeddings / hidden_states / read / inject /
logits); the per-token lockstep and windowing are orchestrated by hybrid.py.

INTERPRETATION FLAGGED FOR CONFIRMATION (see report):
  Injection location = the final hidden state (post-final-norm, pre-lm_head).
  Simplest mechanism that gives an exact γ=0 warm start without hooking
  internal decoder layers. If you want injection at a specific decoder-layer
  residual instead, that's a change — flagged, not guessed.

  What FEEDS the write port (substrate readout vs Recursion state) is a
  hybrid-wiring decision and is intentionally left to hybrid.py — `inject()`
  accepts any (B, T, H_rec) signal. This is the seam I stopped on (see report).
"""
from __future__ import annotations

import torch
import torch.nn as nn

R1_DEFAULT = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


class HybridTransformer(nn.Module):
    def __init__(
        self,
        h_rec: int = 256,
        model_name: str = R1_DEFAULT,
        freeze: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        read_proj_std: float = 0.02,
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM  # lazy: heavy import

        self.model_name = model_name
        self.h_rec = h_rec
        self.freeze = freeze

        self.backbone = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype
        )
        self.D = self.backbone.config.hidden_size

        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # read surface: hidden (D) -> manifold (H_rec)
        self.read_proj = nn.Linear(self.D, h_rec, bias=False)
        nn.init.normal_(self.read_proj.weight, std=read_proj_std)

        # write surface: manifold signal (H_rec) -> hidden (D), γ-gated from 0.
        # write_proj zero-init + γ=0 ⇒ exact warm start (logits == plain R1).
        self.write_proj = nn.Linear(h_rec, self.D, bias=False)
        nn.init.zeros_(self.write_proj.weight)
        self.gamma = nn.Parameter(torch.zeros(1))

    # ---- backbone access --------------------------------------------------
    def embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token embeddings (B, T, D) — available as a substrate token-input
        source if hybrid wires it that way."""
        return self.backbone.get_input_embeddings()(input_ids)

    def hidden_states(
        self,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Final residual-stream hidden states (B, T, D). Backbone is frozen, so
        we run it under no_grad to save memory; gradients still reach the
        trainable read/write surfaces that consume this tensor."""
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            out = self.backbone.model(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )
        return out.last_hidden_state

    # ---- read / write ports ----------------------------------------------
    def read(self, hidden: torch.Tensor) -> torch.Tensor:
        """Transformer readout t = read_proj(hidden), (B, T, H_rec)."""
        return self.read_proj(hidden)

    def inject(self, hidden: torch.Tensor, signal: torch.Tensor) -> torch.Tensor:
        """γ-gated write into the residual stream: hidden + γ · write_proj(signal).
        `signal` is (B, T, H_rec). At init (γ=0, write_proj=0) this is identity."""
        return hidden + self.gamma * self.write_proj(signal)

    def logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """LM head over (possibly injected) hidden states -> (B, T, vocab)."""
        return self.backbone.lm_head(hidden)

    # ---- convenience: plain backbone logits (no injection) ----------------
    def plain_logits(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Reference path with no recurrent injection — equals the γ=0 hybrid
        logits, useful for the warm-start byte-identity check on Vast."""
        return self.logits(self.hidden_states(input_ids=input_ids,
                                              attention_mask=attention_mask))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
