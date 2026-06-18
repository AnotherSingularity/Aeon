"""
aeon.substrate — the recurrent signal-source substrate behind the port.

Substrate selection is **deployment-time configuration**, not an architectural
commitment: `make_substrate(config)` returns a concrete cell that satisfies the
`SubstratePort` contract. Adding another substrate later means adding one file
and one branch here — no refactor of the joiner.

The concrete cells (`matrix_cell`, `vector_cell`) are imported lazily so that
the framework-free port contract (`port`) can be used and tested without torch.
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
from .conformance import (
    verify_substrate,
    ConformanceReport,
    CheckResult,
    SkipCheck,
    make_ast_drift_check,
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
    "verify_substrate",
    "ConformanceReport",
    "CheckResult",
    "SkipCheck",
    "make_ast_drift_check",
]


def make_substrate(config: Mapping[str, Any]) -> SubstratePort:
    """Factory: build the substrate named by ``config["kind"]``.

    config keys:
        kind     : "matrix" | "vector"   (which Aeon cell to instantiate)
        d_in     : per-token input width
        d_state  : readout width (d_s)
        ...      : remaining keys are passed through to the cell constructor

    Cell implementations are imported lazily so importing this package (and the
    port contract) does not require torch.
    """
    cfg = dict(config)
    kind = cfg.pop("kind", None)
    if kind is None:
        raise ValueError("make_substrate: config must include 'kind'")

    if kind == "matrix":
        from .matrix_cell import MatrixStateCell

        return MatrixStateCell(**cfg)
    if kind == "vector":
        from .vector_cell import VectorStateCell

        return VectorStateCell(**cfg)

    raise ValueError(
        f"make_substrate: unknown substrate kind {kind!r} "
        f"(known: 'matrix', 'vector'). Add a new cell file + a branch here to extend."
    )
