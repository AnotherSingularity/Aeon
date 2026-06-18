"""
aeon — three-source contractive architecture (implementation phase, V0.02.02).

ALL forward-path code is Aeon-original: no external-library architecture imports
(no `transformers`) anywhere reachable from HybridModel.forward().

Layout (see docs/RWKV_STUDY.md for the design rationale):

    recursion.py     # Recursion: the σ<1 contractive joiner (Aeon-original)
    transformer.py   # transformer side: Aeon-original Qwen2 backbone + read/write
                     #   surfaces; R1 weights loaded as init (no transformers import)
    substrate/       # the RNN signal source behind the port
        port.py          abstract port contract (read/write/step + capabilities)
        rwkv_cell.py     RWKV-class implementation
        vru_cell.py      candidate contractive-class implementation
        __init__.py      make_substrate() factory
    hybrid.py        # three-source coupling (slow-clock Recursion over the sources)

`transformers` is an optional, non-architecture dependency used only by the
byte-identity gate (tests/test_byte_identity.py) and the training script's
tokenizer — never by the architecture.
"""
