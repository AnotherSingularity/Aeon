# Provenance

These files are a **focused, text-only subset** of
[`BlinkDL/RWKV-LM`](https://github.com/BlinkDL/RWKV-LM), vendored here so the
citations in [`../../docs/RWKV_STUDY.md`](../../docs/RWKV_STUDY.md) resolve
against the exact code that was read.

- **Upstream:** https://github.com/BlinkDL/RWKV-LM
- **License:** Apache-2.0 (see `LICENSE`, copied verbatim from upstream).
- **What is included:** the files studied for the analysis — the v6 demo, the
  training-form `model.py`, the v7 demos (GPT / RNN / fast), the WKV CUDA
  kernels, and the `README.md` / `RWKV-8.md` framing docs.
- **What is excluded:** images (`*.png`), datasets (`*.arrow`, `*.jsonl`),
  tokenizer vocab dumps, model checkpoints, and the older generations
  (v1–v4, v8 scratch) — none of which were needed for the study. Get the full
  tree from upstream.

This subset is unmodified upstream source. All original analysis lives in
`docs/RWKV_STUDY.md`, not here.
