"""
aeon — three-source contractive architecture (implementation phase, v0.02.01).

Layout (see docs/RWKV_STUDY.md for the design rationale):

    recursion.py     # Recursion: the joiner / σ<1 contractive substrate
                     #   (lives in Dylan's package; do not modify — absent here)
    transformer.py   # transformer side (attention-based source)
    substrate/       # the RNN signal source behind the port
        port.py          abstract port contract (read/write/step + capabilities)
        rwkv_cell.py     RWKV-class implementation
        vru_cell.py      candidate contractive-class implementation
        __init__.py      make_substrate() factory
    hybrid.py        # three-source coupling

Only the `substrate/` subsystem is implemented on this branch; `transformer.py`
and `hybrid.py` are documented stubs pending the existing `recursion.py` joiner
and the in-flight coupling decisions.
"""
