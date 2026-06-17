"""
aeon/transformer.py — the transformer side (attention-based source).

STUB. The transformer is one signal source among several (not a privileged
"reasoner"): it contributes to Recursion's σ<1 manifold and Recursion can write
back into it. Per §e-A/B the intended ports are:
  read  : residual-stream hidden states (final + a few mid-depth taps),
          projected into the manifold;
  write : injection into the residual stream — gated KV memory slot preferred
          over a bare additive shift, kept γ-gated from 0 to protect a warm
          start.

Deferred because it depends on the chosen attention backbone (Qwen2-family) and
on the coupling decisions that belong to the implementation phase. Not wired on
this branch.
"""
from __future__ import annotations


class TransformerSide:  # pragma: no cover - stub
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "transformer side not implemented on v0.02.01 — pending backbone "
            "choice and §e read/write coupling decisions"
        )
