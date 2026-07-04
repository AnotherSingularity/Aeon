"""
aeon — a multi-source contractive architecture.

100% Aeon-original, weights and code: random-initialized, trained end-to-end,
with no external library in any forward path.

Layout:

    recursion.py     # Recursion: the σ<1 contractive joiner (Cayley certificate)
    transformer.py   # transformer side: Aeon transformer + read/write surfaces
    substrate/       # the recurrent signal source behind the port
        port.py          abstract port contract (read/write/step + capabilities)
        matrix_cell.py   matrix-state cell (with adaptive feedback control)
        vector_cell.py   vector-state cell (deliberately simple)
        feedback.py      closed-loop load-sensing feedback controller
        conformance.py   verify_substrate()
        __init__.py      make_substrate() factory
    hybrid.py        # multi-source coupling (slow-clock Recursion over the sources)
    tokenizer.py     # AeonTokenizer: Aeon's own from-scratch SentencePiece wrapper
    data.py          # corpus reader (.txt / .jsonl / directory), shared by training
"""
