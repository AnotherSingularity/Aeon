"""
aeon/substrate/port.py — the substrate port.

The substrate port is the architectural artifact from the RWKV study (§d):
a read / write / cadence contract that *every* recurrent substrate satisfies,
so the substrate is an **interface, not a commitment**. Recursion (the joiner)
is written against this port and is substrate-agnostic; which concrete cell sits
behind it is **deployment-time configuration** via `make_substrate()`.

This module is intentionally framework-free — it imports no torch — so the
contract and its capability negotiation can be reasoned about and tested
independently of any concrete (torch) implementation.

Capability tiers
----------------
REQUIRED — every substrate must implement:
    reset(batch_size, device=None)      prepare / zero state
    step(x_t) -> read                   advance one token (per-token cadence)
    read() -> (B, d_state)              current readout, without advancing
    write(drive)                        stage a (B, d_state) drive for next step
  and must declare:
    d_state, d_in                       readout / input widths
    output_bound : float                elementwise bound on step()/read()

  BOUNDED OUTPUT IS A CONTRACT REQUIREMENT. Every value returned by `step()`
  and `read()` must be finite and satisfy `|output| <= output_bound`
  elementwise. Rationale: Recursion's σ<1 certificate yields *system-wide*
  boundedness only if Recursion's inputs are bounded. The port enforces
  input-boundedness on the substrate side, where it is cheap and local — a
  substrate that can emit unbounded readouts can drive the joiner's manifold
  out of its certified regime. Cells realise the bound natively (e.g. a tanh
  wrap, `output_bound = 1.0`) or with an explicit clamp/normalize before return.

OPTIONAL — advertised via `.capabilities()`; the joiner calls `.has(cap)` and
uses these only when present, falling back to the required tier otherwise:
    MATRIX_READ     read_matrix()           rich state read (e.g. RWKV's S)
    DECAY_CONTROL   read_decay()            READ-ONLY decay introspection
    ASSOC_WRITE     assoc_write(k, v, a)    delta-rule association write
    PER_LAYER_READ  read_per_layer()        per-layer readouts

  DECAY IS SUBSTRATE-OWNED. `DECAY_CONTROL` is a *read-only* capability:
  `read_decay()` lets the joiner introspect a substrate's decay (for monitoring
  and capability negotiation) but the joiner never mutates it. The substrate
  owns its decay — VRU returns its fixed geometric scalar; RWKV returns its
  per-channel learned tensor. There is deliberately no decay mutator in the
  port.

A substrate advertises optional capabilities by listing them in the class-level
`CAPABILITIES` frozenset and overriding the corresponding method.
"""
from __future__ import annotations

import abc
from typing import Any, Iterable

# ---- optional-capability identifiers --------------------------------------
MATRIX_READ = "matrix_read"
DECAY_CONTROL = "decay_control"      # read-only introspection (see read_decay)
ASSOC_WRITE = "assoc_write"
PER_LAYER_READ = "per_layer_read"

ALL_CAPABILITIES = frozenset(
    {MATRIX_READ, DECAY_CONTROL, ASSOC_WRITE, PER_LAYER_READ}
)


class CapabilityError(RuntimeError):
    """Raised when an optional port is used on a substrate that does not
    advertise it. The joiner is expected to gate on `.has(cap)` first; this is
    the backstop."""


class SubstratePort(abc.ABC):
    """Abstract substrate port. Concrete cells subclass this (and, when they are
    neural, `torch.nn.Module` alongside it).

    Required-tier contract. Concrete cells must set `d_state` (readout width
    `d_s`), `d_in` (per-token input width), and `output_bound` (the elementwise
    bound their readouts satisfy), and implement the four required methods.
    Optional capabilities are opt-in via `CAPABILITIES`.
    """

    #: optional capabilities this class advertises; override in subclasses
    CAPABILITIES: frozenset = frozenset()

    #: readout width (d_s), input width (d_in), output bound
    d_state: int
    d_in: int
    output_bound: float

    # ---- required tier ----------------------------------------------------
    @abc.abstractmethod
    def reset(self, batch_size: int, device: Any = None) -> None:
        """Prepare / zero internal state for a batch."""

    @abc.abstractmethod
    def step(self, x_t: Any) -> Any:
        """Advance one token. `x_t` is (B, d_in); returns the (B, d_state)
        readout after the step. The readout MUST be finite and bounded by
        `output_bound` elementwise. Defines the substrate's per-token cadence."""

    @abc.abstractmethod
    def read(self) -> Any:
        """Return the current (B, d_state) readout without advancing. MUST be
        finite and bounded by `output_bound` elementwise."""

    @abc.abstractmethod
    def write(self, drive: Any) -> None:
        """Stage a (B, d_state) drive from the joiner, applied at the next
        `step`. The required-tier write port is exactly this vector drive."""

    # ---- capability negotiation -------------------------------------------
    def capabilities(self) -> frozenset:
        return frozenset(self.CAPABILITIES)

    def has(self, capability: str) -> bool:
        return capability in self.CAPABILITIES

    def require(self, capability: str) -> None:
        if not self.has(capability):
            raise CapabilityError(
                f"{type(self).__name__} does not advertise capability "
                f"{capability!r}; advertised: {sorted(self.CAPABILITIES)}"
            )

    # ---- optional tier (default: unsupported) -----------------------------
    def read_matrix(self) -> Any:
        self.require(MATRIX_READ)
        raise NotImplementedError  # pragma: no cover - cells override

    def read_decay(self) -> Any:
        """READ-ONLY decay introspection (see class docstring). Returns the
        substrate's decay descriptor; never mutates it."""
        self.require(DECAY_CONTROL)
        raise NotImplementedError  # pragma: no cover

    def assoc_write(self, k: Any, v: Any, a: Any = None) -> None:
        self.require(ASSOC_WRITE)
        raise NotImplementedError  # pragma: no cover

    def read_per_layer(self) -> Iterable[Any]:
        self.require(PER_LAYER_READ)
        raise NotImplementedError  # pragma: no cover
