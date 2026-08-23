# scripts/colab — free-Colab fluency helpers

Every script here supports the `Aeon_English_Fluency_Colab.ipynb`
notebook. They are Colab-agnostic (run on any CUDA-capable Linux) but
their defaults assume the notebook's Google-Drive layout.

| Script | Purpose |
| ------ | ------- |
| `verify_bundle.py` | Recompute SHA-256 of every file listed in `SHA256_MANIFEST.json`. Halts non-zero on drift. |
| `download_wikitext103.py` | Fetch WikiText-103 raw from the canonical S3 URL; verify byte size (191,984,949) + SHA-256 (`91c00ae2…4a33794`) BEFORE extraction. |
| `env_check.py` | Print GPU model, VRAM, PyTorch, CUDA version; halt if CUDA absent. |
| `benchmark.py` | Short fixed-token fwd+bwd benchmark; prints tokens/hour and estimated Colab sessions. |
| `train_stage.py` | Resumable native Aeon trainer, `--stage stage1` or `--stage stage2`. Saves optimizer/scheduler/RNG/tokens/step per checkpoint; auto-resumes from the latest valid checkpoint on Drive; wall-time cap fires clean halt. |
| `evaluate_and_generate.py` | Perplexity on WikiText valid/test; response-masked loss on `dolly15k_fresh_eval`; raw greedy generations. Verifies `fresh_eval_lock_sha256` before scoring. |
| `build_notebook.py` | Assemble the notebook JSON. |
| `build_bundle.py` | Assemble the Colab zip + SHA-256 manifest. |

## Invariants preserved by these scripts

* Architecture fingerprint A₀ = `sha256:2f895a05…a972f9`
* `total_parameters` = 7,015,366
* `K` = 16
* `MARGIN_H` = 0.02, `MARGIN_C` = 0.02
* Protected P2 SHA-256 = `sha256:962fcd5e…4db9fc3c` (never overwritten)
* Tokenizer SHA-256 = `sha256:064ab6a9…de533481`
* θ immutable during inference

## What these scripts never do

* Call another model, teacher, API, judge, corrector, retrieval system.
* Introduce LoRA, adapters, hypernetworks, fallback language models.
* Rewrite generated text or perform post-generation grammar repair.
* Modify parameters during inference.
* Package Windows outputs.
