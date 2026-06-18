# `aeon.substrate` — the recurrent signal source behind the port

Implements the substrate **port**: the read / write / cadence contract every
substrate cell satisfies, so the substrate is an *interface, not a commitment*.
Cell choice is **deployment-time configuration** via `make_substrate(config)`.

```python
from aeon.substrate import make_substrate
sub = make_substrate({"kind": "matrix", "d_in": 256, "d_state": 256})  # or "vector"
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
| `matrix_cell.py` | **matrix-state cell**: state `(H,N,N)`, per-channel decay, outer-product write; tanh-bounded readout. Advertises `matrix_read`, `decay_control` (read-only), `assoc_write`. |
| `vector_cell.py` | **vector-state cell**: a single state `h` of dim `H`, `h = tanh(W_x x + scalar · W_h h)` with a fixed geometric `scalar`, no gates/carry/clamping. Output is `h` (tanh-bounded). Advertises `decay_control` (read-only). |
| `conformance.py` | `verify_substrate(cell)` — the conformance entry point. |

Two contract points the port enforces:

- **Decay is substrate-owned (read-only).** `decay_control` exposes
  `read_decay()` for the joiner to *introspect* (vector cell: the fixed scalar;
  matrix cell: the per-channel learned tensor). There is no decay mutator.
- **Bounded output is required.** `step()`/`read()` return finite values within
  `output_bound` elementwise (both cells: `1.0`, via tanh). Recursion's σ<1
  certificate gives system-wide boundedness only if its inputs are bounded, so
  the port enforces input-boundedness at the substrate side.

## Conformance — run this when adding a substrate

`verify_substrate(cell)` runs the full port contract and returns a structured
`ConformanceReport` (pass / fail / skip per check, with diagnostics):

```python
from aeon.substrate import make_substrate, verify_substrate
report = verify_substrate(make_substrate({"kind": "vector", "d_in": 256, "d_state": 256}))
assert report.ok, report
```

It covers the required tier (reset/step/read/write shapes, bounded output),
decay read-only, and capability negotiation. Cells register **per-cell
structural checks** via a `CONFORMANCE_CHECKS` class attribute; the vector cell
registers a torch-free AST guard (it must stay a simple single-state recurrence,
distinct from Recursion's contractive machinery) and a torch runtime-structure
check. Torch-free where possible; execution/runtime checks skip cleanly when
torch is unavailable.
