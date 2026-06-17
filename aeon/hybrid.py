"""
aeon/hybrid.py — three-source coupling (Stage-1 hybrid).

⚠️ UNRUN. Written to spec; no torch in the authoring environment. Verified on
Vast (per the agreed plan).

SOURCES (all project into Recursion's σ<1 manifold; none is privileged):
  * substrate  — per-token RNN behind the port (RWKV-class or candidate cell)
  * transformer — frozen Qwen2 backbone (read/write surfaces train)
  * Recursion  — the multi-input contractive joiner (slow clock)

WIRING (relayed answers + HYBRID_DESIGN points 1–4):
  fast clock (per token, within a K-window w; conditioning state = h_{w-1}):
      x_i  = emb_proj(emb_i) + cond_proj(h_{w-1})      # token emb + held slow state
      r_i  = substrate.step(x_i)                        # per-token readout
  slow clock (once at end of window w):
      s_w  = s_proj( mean_i r_i )                       # running mean over the K readouts
      t_w  = transformer.read(hidden)[:, last_i, :]     # K-th transformer readout
      h_w, c_w = recursion.step(s_w, t_w, h_{w-1}.detach(), c_{w-1}.detach())
  output path (Q1 answer = Recursion slow state feeds the write port):
      inject_signal[window w tokens] = h_{w-1}          # broadcast, held across the window
      logits = lm_head( hidden + γ · write_proj(inject_signal) )   # final-hidden inject

DERIVED DESIGN DECISIONS — FLAGGED FOR CONFIRMATION against HYBRID_DESIGN.md
(these are forced by "must train" ∧ "no future leakage" ∧ "slow clock", not
free guesses; confirm before the Vast run):
  (D1) The HELD state conditioning window w is h_{w-1} (previous window's tick
       output) — for BOTH the substrate input (per point 4) AND the transformer
       inject (per the Q1 answer). Causal: h_{w-1} aggregates only past tokens,
       so no within-window future leakage.
  (D2) Truncated BPTT: the recurrence carry into each tick is detached
       (h_{w-1}.detach(), c_{w-1}.detach()) and the substrate state is detached
       at each window boundary (substrate.detach_state()). The conditioning
       state h_{w-1} is NOT detached when used to condition window w, so window
       w's loss backprops one window into window w-1's substrate+recursion —
       this is what gives those params gradient ("gradient flows within a
       window", point 3).
  (D3) Recursion runs in fp32 to protect the σ<margin certificate (Cayley
       solve / svd lack bf16 support); s_w, t_w are cast to fp32 at the tick and
       h is cast back to the compute dtype for projection/inject.
  (D4) Minor: the final window's tick output conditions no later window, so the
       last window's substrate readouts receive no gradient. Accepted for v1.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .substrate import make_substrate
from .recursion import RecursionJoiner
from .transformer import HybridTransformer, R1_DEFAULT


class HybridModel(nn.Module):
    def __init__(
        self,
        h_rec: int = 256,
        K: int = 16,
        model_name: str = R1_DEFAULT,
        substrate: dict | None = None,
        margin_h: float = 0.98,
        margin_c: float = 0.95,
        freeze_backbone: bool = True,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.K = K
        self.h_rec = h_rec

        # transformer side (frozen backbone + trainable read/write surfaces)
        self.transformer = HybridTransformer(
            h_rec=h_rec, model_name=model_name, freeze=freeze_backbone, dtype=dtype
        )
        self.D = self.transformer.D

        # substrate behind the port (deployment-time choice via config)
        sub_cfg = dict(substrate or {"kind": "rwkv", "d_in": h_rec, "d_state": h_rec})
        sub_cfg.setdefault("d_in", h_rec)
        sub_cfg.setdefault("d_state", h_rec)
        self.substrate = make_substrate(sub_cfg)
        self.d_in = self.substrate.d_in
        self.d_state = self.substrate.d_state

        # Recursion joiner (kept fp32 for the certificate; see D3)
        self.recursion = RecursionJoiner(
            h_rec=h_rec, d_substrate=h_rec, d_transformer=h_rec,
            margin_h=margin_h, margin_c=margin_c,
        )

        # hybrid projections (trainable)
        self.emb_proj = nn.Linear(self.D, self.d_in, bias=False)      # token emb -> substrate in
        self.cond_proj = nn.Linear(h_rec, self.d_in, bias=False)      # held slow state -> substrate in
        self.s_proj = nn.Linear(self.d_state, h_rec, bias=False)      # substrate readout -> manifold

    # ---- forward ----------------------------------------------------------
    def forward(self, input_ids, attention_mask=None, labels=None):
        B, T = input_ids.shape
        device = input_ids.device

        emb = self.transformer.embeddings(input_ids)                 # (B,T,D)
        compute_dtype = emb.dtype
        hidden = self.transformer.hidden_states(
            input_ids=input_ids, attention_mask=attention_mask)      # (B,T,D), frozen
        t_all = self.transformer.read(hidden)                        # (B,T,H_rec)
        emb_in = self.emb_proj(emb)                                  # (B,T,d_in)

        self.substrate.reset(B, device)
        h, c = self.recursion.init_state(B, device=device)           # fp32 (D3)

        inject_cols = []
        num_windows = math.ceil(T / self.K)
        for w in range(num_windows):
            start = w * self.K
            end = min((w + 1) * self.K, T)

            h_cond = h                                               # held state for this window (D1)
            cond_in = self.cond_proj(h_cond.to(compute_dtype))       # (B, d_in)

            window_readouts = []
            for i in range(start, end):
                x_i = emb_in[:, i, :] + cond_in                      # token emb + held slow state
                r_i = self.substrate.step(x_i)                       # (B, d_state)
                window_readouts.append(r_i)
                inject_cols.append(h_cond)                           # broadcast (B,H_rec) per token (D1)

            mean_readout = torch.stack(window_readouts, dim=0).mean(dim=0)   # (B, d_state)
            s_w = self.s_proj(mean_readout)                          # (B, H_rec)
            t_w = t_all[:, end - 1, :]                               # (B, H_rec) K-th readout

            # slow-clock tick; truncate carry + substrate state at the boundary (D2)
            h, c = self.recursion.step(
                s_w.float(), t_w.float(), h.detach(), c.detach())
            self.substrate.detach_state()

        inject_signal = torch.stack(inject_cols, dim=1).to(compute_dtype)    # (B,T,H_rec)
        injected = self.transformer.inject(hidden, inject_signal)            # (B,T,D)
        logits = self.transformer.logits(injected)                          # (B,T,V)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        try:
            from transformers.modeling_outputs import CausalLMOutputWithPast
            return CausalLMOutputWithPast(loss=loss, logits=logits)
        except Exception:
            return {"loss": loss, "logits": logits}

    # ---- training helpers -------------------------------------------------
    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def audit(self) -> dict:
        a = self.recursion.audit()
        a["gamma"] = self.transformer.gamma.detach().float().item()
        return a
