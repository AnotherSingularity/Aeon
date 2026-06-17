# RWKV Signal-Source Study

Parallel research work: a study of [RWKV](https://github.com/BlinkDL/RWKV-LM)'s
state-propagation design as **signal-source research** for a **multi-input
contractive architecture**. In that architecture **Recursion is the substrate**
— it has its own state and σ<1 contractive dynamics, and signal sources project
into its manifold through input ports. **RWKV and VRU are candidates for the RNN
signal-source port** (one input); the transformer is another source. The design
is **not limited to two sources** — further inputs plug into Recursion's port
surface without changing its substrate nature.

> **Note on repo location.** This was meant to live in a separate repository.
> The session's GitHub access could not create or fork a new repo (the
> integration lacks repo-creation permission, and forking the upstream was out
> of session scope), so the study lives here, in the repo that was made
> available for it. The content is self-contained.

## Contents

- **[`docs/RWKV_STUDY.md`](docs/RWKV_STUDY.md)** — the analysis. Covers how
  state propagates in RWKV (per-block matrix state, time-mix recurrence,
  per-channel decay, token-shift, the RWKV-7 delta-rule + value-residual), the
  structural contrast with attention/KV-cache, RWKV read as a **candidate RNN
  signal source** (the read/write *ports* it presents to Recursion), the
  an **argued position** on the RNN-source decision (RWKV-7 vs the contractive
  recurrent substrate, with the information asymmetry between them flagged), and
  **positions across** the multi-input substrate design space — including that
  the architecture is not structurally limited to two sources. Positions are
  input to deliberation, not decisions.
- **[`reference/RWKV-LM/`](reference/RWKV-LM/)** — a focused, text-only subset
  of the upstream RWKV-LM source that the analysis cites, so the file/line
  references resolve in-repo. See its `PROVENANCE.md` (upstream is Apache-2.0).

## Reading order

1. `docs/RWKV_STUDY.md` — start here.
2. Follow its citations into `reference/RWKV-LM/` as needed.
