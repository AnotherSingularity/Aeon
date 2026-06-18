"""
aeon/recursion.py — Recursion: the σ<1 contractive joiner (multi-input).

⚠️ UNRUN. Written to spec for the Stage-1 hybrid; no torch in the authoring
environment, so numeric behaviour is verified on Vast (per the agreed plan).

WHAT THIS IS
------------
Recursion is the multi-input contractive substrate the signal sources project
into. It is NOT a signal source and NOT the substrate cell behind the port — it
is the joiner. Its recurrent map carries a hard σ < margin certificate by
construction (Cayley orthogonal × tanh-diagonal-decay, scaled by sigmoid·margin),
so the manifold is a contraction in the Euclidean metric. The certificate is
structural, not statistical: there is no setting of the parameters that violates
σ < margin.

CANONICAL TWO-STATE FORM (chart B), extended to multi-input
-----------------------------------------------------------
    c_next = (1 - λ) · c + λ · tanh(h @ W_cᵀ)                       # delta-decay carry
    h_next = tanh( W_s(s) + W_t(t) [+ W_e(e)] + h @ W_hᵀ + c_next ) # contractive update
with
    W_h = sigmoid(s_h) · MARGIN_H · Cayley(A_h) · diag(tanh(d_h))   ⇒ σ(W_h) < MARGIN_H
    W_c = sigmoid(s_c) · MARGIN_C · Cayley(A_c) · diag(tanh(d_c))   ⇒ σ(W_c) < MARGIN_C
    λ   = sigmoid(log_lambda)
where
    s = substrate readout, t = transformer readout (both at H_rec; see below),
    e = window mean of the ORIGINAL token embeddings (at d_model), projected by
        W_e (d_model → H_rec). Optional, gated by `use_embedding_input` (default
        ON): gives Recursion direct access to raw token-level information at
        integration time. W_e is an input map only — it does NOT carry the
        contraction certificate (only W_h / W_c do), so the σ<margin guarantee
        is unaffected.

CADENCE: `step()` ticks ONCE per call. Recursion does not know about K — the
slow-clock cadence and K-window aggregation are owned by hybrid.py. `step()` is
functional (state passed in / returned) so hybrid can detach at window
boundaries for truncated BPTT.

INTERPRETATIONS FLAGGED FOR CONFIRMATION (see report):
  (1) The carry term `c_next` is placed INSIDE `h_next`'s tanh, per canonical
      chart B ("exact chart-B recurrence" + "canonical delta-decay carry
      update"). The relayed shorthand `h_new = tanh(W_s·s + W_t·t + W_h·h)`
      omitted `+ c_next`; canonical chart B includes it. Implemented canonical.
  (2) `s` and `t` are taken to arrive already projected to H_rec ("both
      pre-projected to H_rec dimension"); W_s, W_t are therefore H_rec→H_rec
      input maps. Their input widths are configurable (default H_rec) so the
      upstream projection can live in the read ports OR here.
"""
from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# spectral utilities
# ---------------------------------------------------------------------------
def sigma_max(W: torch.Tensor) -> float:
    """Largest singular value of W (read-only). svdvals lacks bf16/fp16 CUDA
    support, so cast to fp32 for the audit."""
    Wd = W.detach()
    if Wd.dtype in (torch.bfloat16, torch.float16):
        Wd = Wd.float()
    return torch.linalg.svdvals(Wd).max().item()


def cayley(A: torch.Tensor) -> torch.Tensor:
    """Cayley transform A -> orthogonal Q = (I + S)^{-1}(I - S), S = A - Aᵀ.
    Solve in fp32 (cusolver lacks bf16) and cast back."""
    orig_dtype = A.dtype
    A_f = A.float() if A.dtype in (torch.bfloat16, torch.float16) else A
    S = A_f - A_f.transpose(-2, -1)
    eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device)
    Q = torch.linalg.solve(eye + S, eye - S)
    return Q.to(orig_dtype)


# ---------------------------------------------------------------------------
# Recursion joiner
# ---------------------------------------------------------------------------
class RecursionJoiner(nn.Module):
    def __init__(
        self,
        h_rec: int = 256,
        d_substrate: int | None = None,
        d_transformer: int | None = None,
        d_embedding: int | None = None,
        use_embedding_input: bool = True,
        margin_h: float = 0.98,
        margin_c: float = 0.95,
        learnable_init: bool = True,
    ):
        super().__init__()
        assert 0.0 < margin_h < 1.0 and 0.0 < margin_c < 1.0
        self.H = h_rec
        self.d_substrate = d_substrate or h_rec
        self.d_transformer = d_transformer or h_rec
        self.use_embedding_input = use_embedding_input
        self.d_embedding = d_embedding or h_rec
        self.MARGIN_H = margin_h
        self.MARGIN_C = margin_c

        # Cayley-D parameters for W_h and W_c
        self.A_h = nn.Parameter(0.05 * torch.randn(h_rec, h_rec))
        self.d_h = nn.Parameter(0.5 * torch.randn(h_rec))
        self.s_h = nn.Parameter(torch.tensor(1.5))
        self.A_c = nn.Parameter(0.05 * torch.randn(h_rec, h_rec))
        self.d_c = nn.Parameter(0.5 * torch.randn(h_rec))
        self.s_c = nn.Parameter(torch.tensor(1.5))

        self.log_lambda = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5

        # separate input projections, summed into the tanh update
        self.W_s = nn.Linear(self.d_substrate, h_rec, bias=True)
        self.W_t = nn.Linear(self.d_transformer, h_rec, bias=False)
        # optional third input: window mean of the original embeddings (d_model)
        if use_embedding_input:
            self.W_e = nn.Linear(self.d_embedding, h_rec, bias=False)

        if learnable_init:
            self.h_init = nn.Parameter(torch.zeros(h_rec))
            self.c_init = nn.Parameter(torch.zeros(h_rec))
        else:
            self.register_buffer("h_init", torch.zeros(h_rec))
            self.register_buffer("c_init", torch.zeros(h_rec))

    # ---- weight builders --------------------------------------------------
    def _build(self, A, d, s, margin) -> torch.Tensor:
        Q = cayley(A)
        D = torch.diag(torch.tanh(d))
        scale = torch.sigmoid(s) * margin
        return scale * (Q @ D)

    def W_h_mat(self) -> torch.Tensor:
        return self._build(self.A_h, self.d_h, self.s_h, self.MARGIN_H)

    def W_c_mat(self) -> torch.Tensor:
        return self._build(self.A_c, self.d_c, self.s_c, self.MARGIN_C)

    # ---- state ------------------------------------------------------------
    def init_state(self, batch_size: int, device=None, dtype=None):
        device = device or self.h_init.device
        h = self.h_init.to(device=device, dtype=dtype).expand(batch_size, -1).contiguous()
        c = self.c_init.to(device=device, dtype=dtype).expand(batch_size, -1).contiguous()
        return h, c

    # ---- one tick ---------------------------------------------------------
    def step(self, s: torch.Tensor, t: torch.Tensor, h: torch.Tensor,
             c: torch.Tensor, e: torch.Tensor | None = None):
        """One Recursion tick. s (B, d_substrate), t (B, d_transformer),
        h, c (B, H_rec); e (B, d_embedding) is the window's embedding mean, used
        iff use_embedding_input. Returns (h_next, c_next). Ticks exactly once;
        hybrid.py owns the K-window cadence and may detach (h, c) at boundaries.

            h_next = tanh(W_s·s + W_t·t [+ W_e·e] + h·W_hᵀ + c_next)
        """
        Wh = self.W_h_mat()
        Wc = self.W_c_mat()
        lam = torch.sigmoid(self.log_lambda)
        c_next = (1.0 - lam) * c + lam * torch.tanh(h @ Wc.T)
        pre = self.W_s(s) + self.W_t(t) + h @ Wh.T + c_next
        if self.use_embedding_input:
            if e is None:
                raise ValueError(
                    "RecursionJoiner.step: use_embedding_input=True but e is None"
                )
            pre = pre + self.W_e(e)
        h_next = torch.tanh(pre)
        return h_next, c_next

    # ---- audit hook -------------------------------------------------------
    @torch.no_grad()
    def audit(self) -> dict:
        """σ(W) monitoring. `holds` is True iff both certificates hold."""
        sh = sigma_max(self.W_h_mat())
        sc = sigma_max(self.W_c_mat())
        return {
            "sigma_Wh": sh,
            "sigma_Wc": sc,
            "margin_h": self.MARGIN_H,
            "margin_c": self.MARGIN_C,
            "lambda": torch.sigmoid(self.log_lambda).item(),
            "holds": bool(sh < self.MARGIN_H and sc < self.MARGIN_C),
        }
