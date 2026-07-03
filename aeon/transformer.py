"""
aeon/transformer.py — Aeon's transformer-side signal source.

Aeon-original and random-initialized, end-to-end. No external reference
implementation and no external library in the forward path.

Components (all Aeon-written, to Aeon's own dimensions): RMSNorm, rotary position
embeddings, grouped-query attention, SwiGLU MLP, a pre-norm decoder layer, the
stack, and a tied output head.

Precision choices (empirical, from Aeon's bf16 runs — they keep bf16 training
numerically sound):
  * rotary inv_freq is computed fresh in fp32 every forward (never a buffer that
    a dtype cast could degrade);
  * RMSNorm upcasts to fp32 internally;
  * attention softmax is computed in fp32, then cast back.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# config — Aeon's architecture choices (not inherited from any external model)
# ---------------------------------------------------------------------------
@dataclass
class AeonTransformerConfig:
    vocab_size: int = 128000              # multilingual (top-50 languages); tied I/O embedding
    hidden_size: int = 1024
    intermediate_size: int = 2816
    num_hidden_layers: int = 24
    num_attention_heads: int = 16
    num_key_value_heads: int = 4          # grouped-query attention
    head_dim: int = 64                    # 16 * 64 = 1024
    max_position_embeddings: int = 8192
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    init_std: float = 0.02


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------
class AeonRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dt)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)  # (B,1,T,hd)
    sin = sin.unsqueeze(1)
    return (q * cos) + (_rotate_half(q) * sin), (k * cos) + (_rotate_half(k) * sin)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


class AeonRotary(nn.Module):
    """RoPE cos/sin. inv_freq is computed FRESH in fp32 on every forward — never
    stored — so a dtype cast can never degrade the rope frequencies. Angles are
    fp32; cos/sin are cast to the activation dtype at the end."""

    def __init__(self, cfg: AeonTransformerConfig):
        super().__init__()
        self.head_dim = cfg.head_dim
        self.rope_theta = cfg.rope_theta

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        inv_freq = 1.0 / (self.rope_theta ** (
            torch.arange(0, self.head_dim, 2, dtype=torch.int64, device=x.device).float()
            / self.head_dim))
        inv = inv_freq[None, :, None].expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class AeonAttention(nn.Module):
    def __init__(self, cfg: AeonTransformerConfig):
        super().__init__()
        self.nh = cfg.num_attention_heads
        self.nkv = cfg.num_key_value_heads
        self.hd = cfg.head_dim
        self.n_rep = self.nh // self.nkv
        self.scaling = self.hd ** -0.5
        self.q_proj = nn.Linear(cfg.hidden_size, self.nh * self.hd, bias=cfg.attention_bias)
        self.k_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=cfg.attention_bias)
        self.v_proj = nn.Linear(cfg.hidden_size, self.nkv * self.hd, bias=cfg.attention_bias)
        self.o_proj = nn.Linear(self.nh * self.hd, cfg.hidden_size, bias=False)

    def forward(self, x, cos, sin, attn_mask):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        q, k = _apply_rope(q, k, cos, sin)
        k = _repeat_kv(k, self.n_rep)
        v = _repeat_kv(v, self.n_rep)
        attn = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class AeonMLP(nn.Module):
    def __init__(self, cfg: AeonTransformerConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AeonDecoderLayer(nn.Module):
    def __init__(self, cfg: AeonTransformerConfig):
        super().__init__()
        self.self_attn = AeonAttention(cfg)
        self.mlp = AeonMLP(cfg)
        self.input_layernorm = AeonRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = AeonRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class AeonTransformer(nn.Module):
    """Aeon's transformer stack: embeddings, pre-norm decoder layers, final norm,
    tied output head. Random-initialized; trained from scratch."""

    def __init__(self, cfg: AeonTransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([AeonDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = AeonRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = AeonRotary(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.init_weights()
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def init_weights(self):
        std = self.cfg.init_std
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=std)

    def _causal_mask(self, B, T, dtype, device, attention_mask):
        min_val = torch.finfo(dtype).min
        m = torch.full((T, T), min_val, dtype=dtype, device=device).triu(1)
        m = m[None, None, :, :].expand(B, 1, T, T).clone()
        if attention_mask is not None:
            pad = (1 - attention_mask[:, None, None, :].to(dtype)) * min_val
            m = m + pad
        return m

    def forward_hidden(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                       position_ids=None):
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        B, T, _ = x.shape
        if position_ids is None:
            position_ids = torch.arange(T, device=x.device)[None, :].expand(B, T)
        cos, sin = self.rotary(x, position_ids)
        mask = self._causal_mask(B, T, x.dtype, x.device, attention_mask)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        return self.norm(x)

    def logits(self, hidden):
        return self.lm_head(hidden)


# ---------------------------------------------------------------------------
# transformer side of the hybrid: Aeon transformer + read/write surfaces
# ---------------------------------------------------------------------------
class HybridTransformer(nn.Module):
    def __init__(
        self,
        h_rec: int = 256,
        config: AeonTransformerConfig | None = None,
        freeze: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        read_proj_std: float = 0.02,
    ):
        super().__init__()
        self.cfg = config or AeonTransformerConfig()
        self.h_rec = h_rec
        self.freeze = freeze
        self.D = self.cfg.hidden_size

        self.model = AeonTransformer(self.cfg).to(dtype)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

        # read surface: hidden (D) -> manifold (H_rec)
        self.read_proj = nn.Linear(self.D, h_rec, bias=False)
        nn.init.normal_(self.read_proj.weight, std=read_proj_std)
        # write surface: manifold (H_rec) -> hidden (D), γ-gated from 0.
        # write_proj is random-initialised (NOT zero): with both γ=0 and
        # write_proj=0, each parameter's gradient is proportional to the other,
        # so neither leaves zero and the write path is dead. Random write_proj
        # + γ=0 keeps the injection zero at init while letting γ receive gradient.
        self.write_proj = nn.Linear(h_rec, self.D, bias=False)
        nn.init.normal_(self.write_proj.weight, std=0.02)
        # γ is an fp32 MASTER parameter (a bf16 γ has ULP above the optimizer
        # step near 2^-5, snapping it to 1/32 and freezing it). Declared fp32
        # here; the training script also re-casts it to fp32 after any global
        # dtype cast, which is the load-bearing fix.
        self.gamma = nn.Parameter(torch.zeros(1, dtype=torch.float32))

    # ---- backbone access --------------------------------------------------
    def embeddings(self, input_ids):
        return self.model.embed_tokens(input_ids)

    def hidden_states(self, input_ids=None, inputs_embeds=None, attention_mask=None):
        ctx = torch.no_grad() if self.freeze else torch.enable_grad()
        with ctx:
            return self.model.forward_hidden(
                input_ids=input_ids, inputs_embeds=inputs_embeds,
                attention_mask=attention_mask)

    # ---- read / write ports ----------------------------------------------
    def read(self, hidden):
        return self.read_proj(hidden)

    def inject(self, hidden, signal):
        # fp32 residual add keeps γ's gradient path fp32 from loss back to the
        # parameter. At γ=0 this is identity (hidden.float() + 0, cast back).
        return (hidden.float() + self.gamma * self.write_proj(signal).float()).to(hidden.dtype)

    def logits(self, hidden):
        return self.model.logits(hidden)

    def logits_no_inject(self, input_ids, attention_mask=None):
        """LM logits with no recurrent injection (the γ=0 path), for diagnostics."""
        return self.logits(self.hidden_states(input_ids=input_ids, attention_mask=attention_mask))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
