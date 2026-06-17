# Provenance — RWKV-LM study reference (removed)

This file is the **audit-trail record** for source that was *temporarily*
vendored here during the RWKV study and has since been **removed**.

## What was here

A focused, text-only subset of [`BlinkDL/RWKV-LM`](https://github.com/BlinkDL/RWKV-LM)
— the files read for [`../docs/RWKV_STUDY.md`](../docs/RWKV_STUDY.md): the v6
demo, the training-form `model.py`, the v7 demos (GPT / RNN / fast), the WKV CUDA
kernels, and the `README.md` / `RWKV-8.md` framing docs. It lived at
`reference/RWKV-LM/` and was unmodified upstream source (Apache-2.0).

## Why it was removed

Under the project's **no-external-codebases principle** (strict reading: do not
clone external codebases into ours, even as a study reference — the forward path
is Aeon-original code written from design *understanding* of referenced
architectures, not wrappers or imports). Vendoring upstream source, even
read-only, is inconsistent with that principle, so the subset was deleted.

The study's citations now point at upstream GitHub directly, pinned to a commit
(see below), instead of resolving against an in-repo copy.

## Pointer for the record

- **Upstream:** https://github.com/BlinkDL/RWKV-LM
- **Pinned commit (what the study read):** `bd552d5e6aaaad88196629f7eb8dc8e24a644484`
- **License:** Apache-2.0
- **Browse the studied files at that commit:**
  https://github.com/BlinkDL/RWKV-LM/tree/bd552d5e6aaaad88196629f7eb8dc8e24a644484

All original analysis lives in `docs/RWKV_STUDY.md`; nothing of substance was
lost by removing the vendored copy.
