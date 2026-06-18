"""
aeon/transformer.py — Aeon-original Qwen2-compatible transformer side.

⚠️ UNRUN, and the byte-identity gate is NOT yet executed (no torch / transformers
/ R1 checkpoint in the authoring env). This module is the architecture; the gate
in tests/test_byte_identity.py is what proves it reproduces HF Qwen2 numerically.
Until that gate passes on a GPU box, the R1 warm-start is unproven.

NO-EXTERNAL-ARCHITECTURE PRINCIPLE (Meaning A): this file imports **no
`transformers`**. Every forward-path component — GQA+RoPE attention, SwiGLU MLP,
RMSNorm, the pre-norm decoder layer, the stack, lm_head — is Aeon-written.
`transformers` appears only in the test-time byte-identity check, never in any
import reachable from `HybridModel.forward()`.

R1 weights are training INITIALIZATION, not an architectural dependency:
`config_from_pretrained()` reads the checkpoint's `config.json` (plain JSON) and
`load_r1_weights()` maps the safetensors tensors into Aeon's module hierarchy.

Target config (DeepSeek-R1-Distill-Qwen-1.5B / Qwen2): hidden 1536, 28 layers,
12 q-heads, 2 kv-heads, head_dim 128, intermediate 8960, RMSNorm eps 1e-6,
tied embeddings. Values are read from config.json, not hardcoded.

QWEN2 DETAILS THAT MATTER FOR BYTE-IDENTITY (all replicated below):
  * q/k/v projections HAVE bias; o_proj does NOT (Qwen2-specific).
  * RMSNorm upcasts to fp32, normalises, casts back, then * weight.
  * RoPE: inv_freq = theta^(-arange(0,d,2)/d); emb = cat(freqs, freqs);
    rotate_half = cat(-x2, x1); applied to q,k. cos/sin computed in fp32.
  * attention softmax computed in fp32 then cast back; scale = head_dim**-0.5.
  * GQA: kv heads repeated n_rep = n_q // n_kv.
  * tied lm_head (weight shared with embed_tokens).
Compare against HF with attn_implementation="eager" (this is eager attention).
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

R1_DEFAULT = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


# ---------------------------------------------------------------------------
# config (read from checkpoint config.json — plain JSON, no transformers)
# ---------------------------------------------------------------------------
@dataclass
class AeonQwen2Config:
    vocab_size: int = 151936
    hidden_size: int = 1536
    intermediate_size: int = 8960
    num_hidden_layers: int = 28
    num_attention_heads: int = 12
    num_key_value_heads: int = 2
    head_dim: int = 128
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    tie_word_embeddings: bool = True
    attention_bias: bool = True            # Qwen2: q/k/v carry bias

    @classmethod
    def from_json(cls, path: str) -> "AeonQwen2Config":
        d = json.load(open(path))
        n_heads = d["num_attention_heads"]
        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            intermediate_size=d["intermediate_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=n_heads,
            num_key_value_heads=d.get("num_key_value_heads", n_heads),
            head_dim=d.get("head_dim", d["hidden_size"] // n_heads),
            max_position_embeddings=d.get("max_position_embeddings", 32768),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            rope_theta=d.get("rope_theta", 10000.0),
            tie_word_embeddings=d.get("tie_word_embeddings", True),
            attention_bias=d.get("attention_bias", True),
        )


def config_from_pretrained(checkpoint_dir: str) -> AeonQwen2Config:
    return AeonQwen2Config.from_json(os.path.join(checkpoint_dir, "config.json"))


# ---------------------------------------------------------------------------
# Aeon-original components
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
    q_ = (q * cos) + (_rotate_half(q) * sin)
    k_ = (k * cos) + (_rotate_half(k) * sin)
    return q_, k_


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


class AeonRotary(nn.Module):
    def __init__(self, cfg: AeonQwen2Config):
        super().__init__()
        inv = 1.0 / (cfg.rope_theta ** (
            torch.arange(0, cfg.head_dim, 2, dtype=torch.int64).float() / cfg.head_dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor):
        # position_ids (B,T) -> cos/sin (B,T,head_dim), computed in fp32
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        pos = position_ids[:, None, :].float()
        freqs = (inv @ pos).transpose(1, 2)            # (B, T, hd/2)
        emb = torch.cat((freqs, freqs), dim=-1)        # (B, T, hd)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)


class AeonAttention(nn.Module):
    def __init__(self, cfg: AeonQwen2Config):
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
        attn = torch.matmul(q, k.transpose(2, 3)) * self.scaling     # (B,nh,T,T)
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


class AeonMLP(nn.Module):
    def __init__(self, cfg: AeonQwen2Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class AeonDecoderLayer(nn.Module):
    def __init__(self, cfg: AeonQwen2Config):
        super().__init__()
        self.self_attn = AeonAttention(cfg)
        self.mlp = AeonMLP(cfg)
        self.input_layernorm = AeonRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = AeonRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, x, cos, sin, attn_mask):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, attn_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class AeonQwen2Model(nn.Module):
    """Aeon-original Qwen2 stack. Module/param names mirror HF so the R1
    safetensors map in by stripping the leading 'model.' prefix."""

    def __init__(self, cfg: AeonQwen2Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([AeonDecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = AeonRMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.rotary = AeonRotary(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

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
# weight loader (R1 safetensors -> Aeon module hierarchy)
# ---------------------------------------------------------------------------
def _safetensors_shards(checkpoint_dir: str):
    idx = os.path.join(checkpoint_dir, "model.safetensors.index.json")
    if os.path.exists(idx):
        files = sorted(set(json.load(open(idx))["weight_map"].values()))
        return [os.path.join(checkpoint_dir, f) for f in files]
    single = os.path.join(checkpoint_dir, "model.safetensors")
    if os.path.exists(single):
        return [single]
    return sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))


def load_r1_weights(model: "AeonQwen2Model", checkpoint_dir: str) -> dict:
    """Map R1's HF Qwen2 safetensors into the Aeon stack. HF keys carry a
    'model.' prefix on the backbone and a top-level 'lm_head.weight'; Aeon's
    hierarchy drops the prefix. Returns a summary dict."""
    from safetensors.torch import load_file  # I/O only, not architecture

    raw = {}
    for f in _safetensors_shards(checkpoint_dir):
        raw.update(load_file(f))

    aeon_sd = {}
    for k, v in raw.items():
        nk = k[len("model."):] if k.startswith("model.") else k
        aeon_sd[nk] = v

    missing, unexpected = model.load_state_dict(aeon_sd, strict=False)
    # tied lm_head: checkpoints with tie_word_embeddings usually omit lm_head.weight
    if "lm_head.weight" not in aeon_sd and model.cfg.tie_word_embeddings:
        model.lm_head.weight = model.embed_tokens.weight
    # `rotary.inv_freq` is a non-persistent buffer; its absence is expected.
    missing = [m for m in missing if not m.endswith("rotary.inv_freq")]
    return {"loaded": len(aeon_sd), "missing": missing, "unexpected": list(unexpected)}


# ---------------------------------------------------------------------------
# the hybrid transformer side (Aeon backbone + read/write surfaces)
# ---------------------------------------------------------------------------
class HybridTransformer(nn.Module):
    def __init__(
        self,
        h_rec: int = 256,
        config: AeonQwen2Config | None = None,
        freeze: bool = True,
        dtype: torch.dtype = torch.bfloat16,
        read_proj_std: float = 0.02,
    ):
        super().__init__()
        self.cfg = config or AeonQwen2Config()
        self.h_rec = h_rec
        self.freeze = freeze
        self.D = self.cfg.hidden_size

        self.model = AeonQwen2Model(self.cfg).to(dtype)
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

        # read surface: hidden (D) -> manifold (H_rec)
        self.read_proj = nn.Linear(self.D, h_rec, bias=False)
        nn.init.normal_(self.read_proj.weight, std=read_proj_std)
        # write surface: manifold (H_rec) -> hidden (D), γ-gated from 0 (warm start)
        self.write_proj = nn.Linear(h_rec, self.D, bias=False)
        nn.init.zeros_(self.write_proj.weight)
        self.gamma = nn.Parameter(torch.zeros(1))

    # ---- weight init from R1 ---------------------------------------------
    def load_pretrained(self, checkpoint_dir: str) -> dict:
        return load_r1_weights(self.model, checkpoint_dir)

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
        return hidden + self.gamma * self.write_proj(signal)

    def logits(self, hidden):
        return self.model.logits(hidden)

    def plain_logits(self, input_ids, attention_mask=None):
        return self.logits(self.hidden_states(input_ids=input_ids, attention_mask=attention_mask))

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
