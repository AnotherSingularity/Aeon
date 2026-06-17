"""
aeon.substrate — the RNN signal-source substrate behind the port (§d).

Substrate selection is **deployment-time configuration**, not an architectural
commitment: `make_substrate(config)` returns a concrete cell that satisfies the
`SubstratePort` contract. Adding a third substrate class later means adding one
file and one branch here — no refactor of the joiner.

The concrete cells (`rwkv_cell`, `vru_cell`) are imported lazily so that the
framework-free port contract (`port`) can be used and tested without torch.
"""
from __future__ import annotations

from typing import Any, Mapping

from .port import (
    SubstratePort,
    CapabilityError,
    MATRIX_READ,
    DECAY_CONTROL,
    ASSOC_WRITE,
    PER_LAYER_READ,
    ALL_CAPABILITIES,
)

__all__ = [
    "SubstratePort",
    "CapabilityError",
    "MATRIX_READ",
    "DECAY_CONTROL",
    "ASSOC_WRITE",
    "PER_LAYER_READ",
    "ALL_CAPABILITIES",
    "make_substrate",
]


def make_substrate(config: Mapping[str, Any]) -> SubstratePort:
    """Factory: build the substrate named by ``config["kind"]``.

    config keys:
        kind     : "rwkv" | "vru"   (which design archetype to instantiate)
        d_in     : per-token input width
        d_state  : readout width (d_s)
        ...      : remaining keys are passed through to the cell constructor

    Substrate-specific implementations are imported lazily so importing this
    package (and the port contract) does not require torch.
    """
    cfg = dict(config)
    kind = cfg.pop("kind", None)
    if kind is None:
        raise ValueError("make_substrate: config must include 'kind'")

    if kind == "rwkv":
        from .rwkv_cell import RWKVCell

        return RWKVCell(**cfg)
    if kind == "vru":
        from .vru_cell import VRUCell

        return VRUCell(**cfg)

    raise ValueError(
        f"make_substrate: unknown substrate kind {kind!r} "
        f"(known: 'rwkv', 'vru'). Add a new cell file + a branch here to extend."
    )
