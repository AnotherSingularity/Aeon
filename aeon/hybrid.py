"""
aeon/hybrid.py — three-source coupling.

STUB. Wires the signal sources (the substrate behind the port + the transformer
side) into Recursion, the multi-input σ<1 contractive joiner. The joiner is
written against `substrate.port.SubstratePort` and is substrate-agnostic, so the
substrate is selected at runtime:

    from aeon.substrate import make_substrate
    substrate = make_substrate(config["substrate"])   # "rwkv" | "vru" | ...

Deferred because it depends on the existing `aeon/recursion.py` joiner (Dylan's
package; do not modify — not present on this branch) and on the §e coupling
choices (read/write port shape, sampling cadence, loop topology) that the
implementation phase produces in code + tests. Not wired on this branch.
"""
from __future__ import annotations


class Hybrid:  # pragma: no cover - stub
    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "three-source coupling not implemented on v0.02.01 — pending the "
            "recursion.py joiner and §e coupling decisions"
        )
