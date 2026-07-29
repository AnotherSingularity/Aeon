# Aeon V0.02.03 — Primary-Training Operations Guide

Covers §13.3 launch/recovery and §13.4 capability milestones. The primary config
is `configs/aeon_350m_primary.yaml`; the diagnostic tool is `scripts/diagnose.py`.

## 1. Whole-model parameter accounting (§13.2)

Source of truth: `docs/e6_parameter_accounting.json` — regenerated whenever the
primary config changes. Current numbers at `configs/aeon_350m_primary.yaml`:

| component | trainable | fp32 bytes |
|---|---:|---:|
| transformer (embedding included, LM head tied) | **346.08 M** | 1384.3 MB |
| substrate (matrix cell + adaptive feedback controller) | 1.57 M | 6.3 MB |
| Recursion (contractive joiner + certificate params) | 1.58 M | 6.3 MB |
| emb_proj (Aeon-native embedding projection to substrate) | 0.52 M | 2.1 MB |
| cond_proj (Recursion broadcast → substrate cond) | 0.26 M | 1.0 MB |
| s_proj (substrate readout → Recursion input) | 0.26 M | 1.0 MB |
| **P_total** | **350.28 M** | 1401.1 MB |

Optimiser overhead (AdamW, 2 fp32 moments): ~2.80 GB.
Checkpoint estimate (bf16 run): ~700 MB per save (plus `.prev` retention).

DO NOT report `transformer` alone as the Aeon parameter budget — the primary
number is 350.28 M. The E6 accounting JSON is the durable record.

## 2. Launch and recovery commands (§13.3)

Before any command, populate the two placeholders in
`configs/aeon_350m_primary.yaml`:

```yaml
data:
  tokenizer: /path/to/tokenizer/aeon.model   # from scripts/train_tokenizer.py
  corpus: /path/to/aeon_corpus/              # dir | .txt | .jsonl
```

Never commit live paths — the tokenizer identity is part of checkpoint
compatibility and reject-incompatible-metadata refuses a swap after resume.

### 2.1 Fresh training

```bash
python scripts/train.py --config configs/aeon_350m_primary.yaml
```

Emits:
- `runs/aeon_350m_primary/ckpt_<step>.pt` (+ `.sha256` sidecar, `.prev` retention)
- `runs/aeon_350m_primary/metrics.jsonl` (JSONL: parameter accounting, always-on, sampled)

### 2.2 Resume from the latest valid checkpoint

The primary config sets `train.resume: true`, so the same command auto-selects
the latest checkpoint:

```bash
python scripts/train.py --config configs/aeon_350m_primary.yaml
```

Under the hood: `strict_load` runs the metadata gate before touching state.

### 2.3 Resume from a specific checkpoint

Edit `train.out_dir` in a copy of the config to point at the checkpoint's
directory, remove any newer siblings, and re-launch. (A future CLI flag can
select a checkpoint explicitly; for now the "latest under out_dir" contract is
the interface.)

### 2.4 Offline diagnostics

```bash
# Every subcommand; report next to the checkpoint
python scripts/diagnose.py --config configs/aeon_350m_primary.yaml \
    --ckpt runs/aeon_350m_primary/ckpt_1000.pt --subcommand all

# Just certificate + gradient probe
python scripts/diagnose.py --config configs/aeon_350m_primary.yaml \
    --ckpt runs/aeon_350m_primary/ckpt_1000.pt --subcommand certificate
python scripts/diagnose.py --config configs/aeon_350m_primary.yaml \
    --ckpt runs/aeon_350m_primary/ckpt_1000.pt --subcommand gradients --bound 512

# English-continuation probe (with a real tokenizer)
python scripts/diagnose.py --config configs/aeon_350m_primary.yaml \
    --ckpt runs/aeon_350m_primary/ckpt_1000.pt --subcommand probes \
    --tokenizer tokenizer/aeon.model \
    --prompt "Aeon is" --prompt "The system"
```

### 2.5 Generation-only test

```bash
python scripts/infer.py --config configs/aeon_350m_primary.yaml \
    --ckpt runs/aeon_350m_primary/ckpt_1000.pt \
    --tokenizer tokenizer/aeon.model \
    --prompt "Aeon" --max-new-tokens 64
```

### 2.6 Integrity verification (checkpoint alone)

```bash
python -c "
from aeon.checkpoint import strict_load
import yaml
mcfg = yaml.safe_load(open('configs/aeon_350m_primary.yaml'))['model']
blob = strict_load('runs/aeon_350m_primary/ckpt_1000.pt',
                   expected_model_config=mcfg)
print('metadata OK:', {k: blob['metadata'][k] for k in ('step','K','patch_manifest_version','source_commit')})
"
```

## 3. Recovery procedures (§13.3)

| situation | procedure |
|---|---|
| **Power interruption** | Re-run 2.2. `atomic_save` guarantees the prior checkpoint survives an interrupted save; `.prev` is the safety net. Resume auto-selects the latest valid file. |
| **Process crash** | Same as 2.2. If the crash left a `.ckpt.tmp.<...>` file, `atomic_save` will clean it up on next attempt. Manual sweep of `runs/*.tmp.*` is safe. |
| **Invalid / incompatible checkpoint** | `strict_load` refuses with a named reason (K mismatch, vocab mismatch, sha256 mismatch, patch_manifest_version drift). Do NOT bypass — either fix the config to match the checkpoint OR delete the checkpoint and drop back to `.prev`. |
| **Low disk space** | Deletes older `ckpt_<step>.pt` files (keep the latest two + the `.prev` of each) before continuing. Watch `metrics.jsonl` growth; rotate weekly. |
| **Memory pressure** | Reduce `train.batch_size` in the config; if that is not sufficient, escalate to a fresh session on a machine with more RAM. Do NOT touch `K`, precision policy, or certificate. |
| **Certificate failure** | Stop immediately (§16.2 fail-closed). Run `scripts/diagnose.py --subcommand certificate` on the last-known-good checkpoint. Investigate whether a preservation invariant broke. Do NOT relax the margin to pass. |
| **Non-finite values** | Also stop-closed. Preserve the offending checkpoint (do not save over it), run `diagnose.py --subcommand gradients`, then follow §9 heavy-debug escalation — with a debug authorization record. |
| **Unexpected slowdown** | Compare `metrics.jsonl` sampled `phase_s` timings with historical baseline. If the sampled path's overhead is drifting, temporarily lower `train.sample_every`. Do NOT disable observability during a live run without recording a debug-authorization note. |

## 4. Capability milestones (§13.4)

Recorded at each checkpoint via `scripts/diagnose.py --subcommand probes` with
a fixed prompt set. Milestones (from cheapest to strongest):

1. **Basic token and word continuation** — the greedy path produces plausible
   next tokens (not `<pad>`, not `<unk>`, non-degenerate).
2. **Grammatical local continuation** — 4-8 token continuations parse as
   grammatical fragments.
3. **Short coherent English response** — 16-32 token continuations answer the
   prompt semantically.
4. **Multi-sentence continuity** — 32-64 token continuations maintain topic.
5. **Context retention** — the model refers back to details from earlier in
   the prompt.
6. **Long-range continuity** — the model tracks state across ≥ K tokens (this
   is where the Recursion path is the differentiator).
7. **Stable generation after resume** — a resumed checkpoint produces
   qualitatively identical continuations to the pre-save one on the same prompt.

Milestones 1-3 are testable on the sanity run (~10 M param proxy). Milestones
4-7 need the primary run. A "smoke-test model" (milestones 1-2 only) MUST NOT
be reported as functionally trained.

## 5. Efficiency-claim boundaries (per §17)

Current framing while primary training is planned but not yet executed:

> Aeon is architecturally designed for bounded long-range integration through
> two parallel streams and a contractive slow-clock Recursion mechanism. Its
> efficiency is being measured at the current implementation scale on
> laptop-class CPU hardware.

After successful small-proxy comparisons (§14):

> Aeon is architecturally efficient by design and has demonstrated measured
> efficiency at small scale under matched or approximately matched CPU
> experiments.

The following claims are OUT OF BOUNDS at every current stage:

- Frontier superiority
- Proven universal compute efficiency
- Proven full-scale superiority
- Better performance than all transformers / all recurrent models
- Scaling results not actually measured
- Energy efficiency without energy measurements
- FLOP efficiency based solely on static operation estimates

## 6. E6 exit gate — checklist

- [x] Primary config exists at `configs/aeon_350m_primary.yaml` and is
      immutable per campaign version.
- [x] Whole-model parameter accounting reported (350.28 M) — not transformer-only.
- [x] Fresh-start command documented and verified via E5 scenario 1-3.
- [x] Resume command documented and verified via E5 scenarios 6-7.
- [x] Recovery procedures listed for every §13.3 situation.
- [x] Capability milestones defined so a smoke-test model cannot be reported
      as functionally trained.
- [x] Efficiency claims bounded to evidence at every stage.

**E6 exit gate: PASS.**
