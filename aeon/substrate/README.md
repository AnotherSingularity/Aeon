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
| `rwkv_cell.py` | **RWKV-class** cell (Aeon-original): matrix state `(H,N,N)`, per-channel decay, outer-product write. Advertises `matrix_read`, `decay_control`, `assoc_write`. |
| `vru_cell.py` | **candidate contractive-class** cell (Aeon-original, *provisional*): spectral-norm-bounded `σ<1` recurrence + EMA carry. Advertises `decay_control`. |

Both cells are written from design understanding of the archetypes — **no
external package is wrapped or imported** (the no-external-codebases principle).

Adding a substrate later = add one `*_cell.py` + one branch in
`make_substrate()`. The joiner never changes — it is written against `port.py`.

**Status (branch v0.02.01):** port contract + factory + both cells + conformance
tests. The contract tests run without torch; the cell numeric tests require
torch (skip cleanly otherwise). `transformer.py` and `hybrid.py` are stubs
pending the existing `recursion.py` joiner and the §e coupling decisions.
