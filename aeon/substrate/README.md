# `aeon.substrate` — the RNN signal source behind the port

Implements the substrate **port** from the study (`docs/RWKV_STUDY.md` §d): the
read / write / cadence contract every recurrent substrate satisfies, so the
substrate is an *interface, not a commitment*. Substrate choice is
**deployment-time configuration** via `make_substrate(config)`.

```python
from aeon.substrate import make_substrate
sub = make_substrate({"kind": "rwkv", "d_in": 512, "d_state": 256})  # or "vru"
sub.reset(batch_size)
read = sub.step(x_t)                 # (B, d_state); per-token cadence
sub.write(drive)                     # stage a (B, d_state) joiner drive
if sub.has("matrix_read"):           # optional, negotiated
    S = sub.read_matrix()
```

| file | role |
|---|---|
| `port.py` | framework-free `SubstratePort` ABC + capability tiers (REQUIRED + OPTIONAL). Imports no torch — the contract is testable without it. |
| `__init__.py` | exports + `make_substrate()` factory (lazy-imports cells). |
| `rwkv_cell.py` | **RWKV-class** cell (Aeon-original): matrix state `(H,N,N)`, per-channel decay, outer-product write; tanh-bounded readout. Advertises `matrix_read`, `decay_control` (read-only), `assoc_write`. |
| `vru_cell.py` | **candidate** cell (Aeon-original), disclosed spec: a single state `h` of dim `H`, recurrence `h = tanh(W_x x + scalar · W_h h)` with a fixed geometric `scalar`, no gates/carry/clamping. Output is `h` (tanh-bounded). Advertises `decay_control` (read-only). |

Both cells are written from design understanding of the archetypes — **no
external package is wrapped or imported** (the no-external-codebases principle).

Two contract points the port enforces:

- **Decay is substrate-owned (read-only).** `decay_control` exposes
  `read_decay()` for the joiner to *introspect* (VRU: the fixed scalar; RWKV:
  the per-channel learned tensor). There is no decay mutator.
- **Bounded output is required.** `step()`/`read()` return finite values within
  `output_bound` elementwise (both cells: `1.0`, via tanh). Rationale:
  Recursion's σ<1 certificate gives system-wide boundedness only if its inputs
  are bounded, so the port enforces input-boundedness at the substrate side.

A structural test (`test_vru_no_recursion_class_mechanisms`) parses `vru_cell.py`
and fails if Recursion-class mechanisms (spectral norm, carry/EMA, gates,
clamping) reappear — a durable guard against drift.

Adding a substrate later = add one `*_cell.py` + one branch in
`make_substrate()`. The joiner never changes — it is written against `port.py`.

**Status (branch v0.02.01):** port contract + factory + both cells + conformance
tests. The contract tests run without torch; the cell numeric tests require
torch (skip cleanly otherwise). `transformer.py` and `hybrid.py` are stubs
pending the existing `recursion.py` joiner and the §e coupling decisions.
