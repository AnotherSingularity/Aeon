# RWKV State-Propagation Study

Parallel research work: a study of [RWKV](https://github.com/BlinkDL/RWKV-LM)'s
state-propagation design, used as a reference for understanding Aeon's own
architectural information flow.

> **Note on repo location.** This was meant to live in a separate repository.
> The session's GitHub access could not create or fork a new repo (the
> integration lacks repo-creation permission, and forking the upstream was out
> of session scope), so the study lives here, in the repo that was made
> available for it. The content is self-contained.

## Contents

- **[`docs/RWKV_STUDY.md`](docs/RWKV_STUDY.md)** — the analysis. Covers how
  state propagates in RWKV (per-block matrix state, time-mix recurrence,
  per-channel decay, token-shift, the RWKV-7 delta-rule + value-residual), the
  structural contrast with attention/KV-cache, where Aeon currently sits in the
  taxonomy ("transformer with sidecar recursion"), and open questions for
  Aeon's next-generation design.
- **[`reference/RWKV-LM/`](reference/RWKV-LM/)** — a focused, text-only subset
  of the upstream RWKV-LM source that the analysis cites, so the file/line
  references resolve in-repo. See its `PROVENANCE.md` (upstream is Apache-2.0).

## Reading order

1. `docs/RWKV_STUDY.md` — start here.
2. Follow its citations into `reference/RWKV-LM/` as needed.
