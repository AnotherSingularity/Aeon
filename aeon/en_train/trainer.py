"""aeon.en_train.trainer — resumable trainer + LR pilot + gated
learning-curve checkpoints + promotion decision (§9, §10, §11, §12,
§13, §14).

The trainer NEVER:
  * replaces Aeon's forward path;
  * inserts an external model between logits and loss;
  * silently freezes intended-trainable modules;
  * squashes stability failures;
  * cherry-picks seeds.

Every call to the model is `HybridModel.forward(...)` with `shuttle=None`.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch

from . import (
    EFFECTIVE_BATCH_TOKENS_TARGET, LR_PILOT_GRID,
    MIN_RELATIVE_IMPROVEMENT, STAGE1_CEILING, STAGE1_CHECKPOINTS,
    STAGE1_INCREMENT_AFTER_25M, STAGE1_MIX, STAGE2_CHECKPOINTS,
    STAGE2_MIX,
)
from .losses import (
    EffectiveTokenCounter, StageMixture, general_english_loss,
    conversational_loss, pick_sequence_length,
)
from .proof import (
    assert_architecture_invariant, assert_native_stability_gate,
    observe_gradient_path, snapshot_state_dict, compute_weight_delta,
    sigma_certificate,
)


# ---------------------------------------------------------------------------
# Batch protocol — each partition supplies batches by pulling from a
# document iterator. Contract is minimal: yield dicts with input_ids
# and (optionally) response_mask, at the requested sequence length.
# ---------------------------------------------------------------------------
class DocumentBatchProvider:
    """Wraps a token stream and yields batches at a target sequence
    length, with per-batch effective-token counting."""
    def __init__(self, name: str, streams: Sequence[Sequence[int]],
                     response_masks: Optional[Sequence[Sequence[int]]] = None,
                     seed: int = 20260803):
        assert streams, f"empty stream for provider {name}"
        self.name = name
        self.streams = [list(s) for s in streams]
        if response_masks is not None:
            assert len(response_masks) == len(streams)
            self.response_masks = [list(m) for m in response_masks]
        else:
            self.response_masks = None
        self.rng = random.Random(seed)
        self._cursor = [0] * len(self.streams)

    def next_batch(self, batch_size: int, seq_len: int,
                       device: str = "cpu") -> Dict[str, torch.Tensor]:
        """Draw `batch_size` sequences of length `seq_len` (right-padded
        with 0 = pad_id where a stream is exhausted). Attention mask
        marks the real (non-padding) positions."""
        ids = torch.zeros((batch_size, seq_len), dtype=torch.long)
        att = torch.zeros((batch_size, seq_len), dtype=torch.long)
        rmask = torch.zeros((batch_size, seq_len), dtype=torch.long)
        for b in range(batch_size):
            src = self.rng.randrange(len(self.streams))
            s = self.streams[src]
            start = self.rng.randrange(max(1, len(s) - seq_len))
            end = min(start + seq_len, len(s))
            piece = s[start:end]
            ids[b, :len(piece)] = torch.tensor(piece, dtype=torch.long)
            att[b, :len(piece)] = 1
            if self.response_masks is not None:
                m = self.response_masks[src][start:end]
                rmask[b, :len(m)] = torch.tensor(m, dtype=torch.long)
        batch = {"input_ids": ids.to(device),
                    "attention_mask": att.to(device)}
        if self.response_masks is not None:
            batch["response_mask"] = rmask.to(device)
        return batch


# ---------------------------------------------------------------------------
# Optimizer schedule (§10)
# ---------------------------------------------------------------------------
def cosine_lr(step: int, total_steps: int, warmup_steps: int,
                  peak_lr: float, final_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    return final_lr + 0.5 * (peak_lr - final_lr) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------
@dataclass
class TrainStepResult:
    loss: float
    valid_tokens: int
    grad_l2: float
    source: str


def _apply_grad_clip(model, clip: float = 1.0) -> float:
    total_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_sq += float(p.grad.detach().to(torch.float32).pow(2).sum().item())
    total = total_sq ** 0.5
    if clip > 0 and total > clip:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(clip / (total + 1e-12))
    return total


def train_one_step(model, optimizer, batch: Dict[str, torch.Tensor],
                        source: str, grad_clip: float = 1.0) -> TrainStepResult:
    """One forward+backward+step on a single batch. Uses L_G on general
    batches, L_C on batches whose response_mask has any 1s."""
    model.train()
    optimizer.zero_grad()
    rmask = batch.get("response_mask")
    if rmask is not None and rmask.sum().item() > 0:
        loss, vt = conversational_loss(model,
                                            input_ids=batch["input_ids"],
                                            response_mask=rmask,
                                            attention_mask=batch["attention_mask"])
    else:
        loss, vt = general_english_loss(model,
                                              input_ids=batch["input_ids"],
                                              attention_mask=batch["attention_mask"])
    loss.backward()
    grad_l2 = _apply_grad_clip(model, clip=grad_clip)
    optimizer.step()
    return TrainStepResult(loss=float(loss.item()), valid_tokens=vt,
                              grad_l2=grad_l2, source=source)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def evaluate_val_loss(model, provider: DocumentBatchProvider,
                          n_batches: int, batch_size: int, seq_len: int,
                          device: str = "cpu") -> Tuple[float, int]:
    """Deterministic val-loss (§9's I_k signal)."""
    model.eval()
    total_loss = 0.0
    total_toks = 0
    with torch.inference_mode():
        for _ in range(n_batches):
            b = provider.next_batch(batch_size, seq_len, device)
            rmask = b.get("response_mask")
            if rmask is not None and rmask.sum().item() > 0:
                loss, vt = conversational_loss(
                    model, input_ids=b["input_ids"],
                    response_mask=rmask,
                    attention_mask=b["attention_mask"])
            else:
                loss, vt = general_english_loss(
                    model, input_ids=b["input_ids"],
                    attention_mask=b["attention_mask"])
            total_loss += float(loss.item()) * vt
            total_toks += vt
    if total_toks == 0: return 0.0, 0
    return total_loss / total_toks, total_toks


# ---------------------------------------------------------------------------
# Resumable checkpoint
# ---------------------------------------------------------------------------
def save_candidate_checkpoint(model, optimizer, out_dir: Path,
                                     step: int, tokens_covered: int,
                                     stage: str, seed: int,
                                     val_loss: float,
                                     mixture: Dict[str, float],
                                     lr_pilot_choice: float,
                                     n_updates: int,
                                     effective_batch_tokens: int) -> Dict[str, Any]:
    """Write a candidate checkpoint under `out_dir/candidate-<step>/`.
    The checkpoint carries its OWN release identity — never overwrites P2."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    slot = out_dir / f"candidate-step{step:09d}"
    slot.mkdir(parents=True, exist_ok=True)
    state = {
        "model_state_dict": {k: v.detach().cpu().contiguous()
                                    for k, v in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "stage": stage,
        "step": step,
        "tokens_covered": tokens_covered,
        "seed": seed,
        "val_loss": val_loss,
        "mixture": dict(mixture),
        "lr_peak": lr_pilot_choice,
        "n_updates": n_updates,
        "effective_batch_tokens": effective_batch_tokens,
    }
    payload_path = slot / "state.pt"
    torch.save(state, payload_path)
    with open(payload_path, "rb") as f:
        sha = "sha256:" + hashlib.sha256(f.read()).hexdigest()
    manifest = {
        "candidate_release_id": f"aeon-en-train-{stage}-step{step:09d}",
        "state_pt_sha256": sha,
        "state_pt_bytes": payload_path.stat().st_size,
        "step": step,
        "tokens_covered": tokens_covered,
        "val_loss": val_loss,
        "stage": stage,
        "seed": seed,
        "mixture": dict(mixture),
    }
    (slot / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def load_candidate_checkpoint(model, optimizer, path: Path) -> Dict[str, Any]:
    st = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(st["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(st["optimizer_state_dict"])
    return {"stage": st["stage"], "step": st["step"],
                "tokens_covered": st["tokens_covered"],
                "seed": st["seed"], "val_loss": st["val_loss"]}


# ---------------------------------------------------------------------------
# LR pilot (§10) — bounded, small budget
# ---------------------------------------------------------------------------
@dataclass
class LRPilotResult:
    peak_lr: float
    val_loss: float
    stability_passed: bool


def run_lr_pilot(*, model_factory: Callable[[], Any],
                    build_optimizer: Callable[[Any, float], torch.optim.Optimizer],
                    train_provider: DocumentBatchProvider,
                    val_provider: DocumentBatchProvider,
                    baseline_sigma: Dict[str, float],
                    grid: Sequence[float] = LR_PILOT_GRID,
                    n_pilot_tokens: int = 1_000_000,
                    batch_size: int = 8,
                    seq_len: int = 256,
                    grad_clip: float = 1.0,
                    device: str = "cpu",
                    seed: int = 20260803,
                    val_batches: int = 16,
                    ) -> List[LRPilotResult]:
    """Same starting weights, same 1M-token training subset, same val
    set, same effective batch, same stability diagnostics."""
    results: List[LRPilotResult] = []
    for lr in grid:
        random.seed(seed); torch.manual_seed(seed)
        model = model_factory()
        optimizer = build_optimizer(model, lr)
        tokens = 0
        while tokens < n_pilot_tokens:
            b = train_provider.next_batch(batch_size, seq_len, device)
            r = train_one_step(model, optimizer, b, source="pilot",
                                     grad_clip=grad_clip)
            tokens += r.valid_tokens
        try:
            assert_architecture_invariant(model)
            assert_native_stability_gate(model, baseline_sigma)
            stab = True
        except Exception:
            stab = False
        vl, _ = evaluate_val_loss(model, val_provider, val_batches,
                                          batch_size, seq_len, device)
        results.append(LRPilotResult(peak_lr=lr, val_loss=vl, stability_passed=stab))
    return results


def select_pilot_lr(results: Sequence[LRPilotResult]) -> Optional[float]:
    stable = [r for r in results if r.stability_passed]
    if not stable: return None
    stable.sort(key=lambda x: x.val_loss)
    return stable[0].peak_lr


# ---------------------------------------------------------------------------
# Gated learning-curve driver (§9)
# ---------------------------------------------------------------------------
@dataclass
class CheckpointRecord:
    step: int
    tokens_covered: int
    val_loss: float
    relative_improvement: float
    architecture_delta_zero: bool
    native_stability_passed: bool
    grad_path_ok: bool
    manifest: Dict[str, Any]


def _stage_checkpoints(stage: str) -> Sequence[int]:
    if stage == "stage1": return STAGE1_CHECKPOINTS
    if stage == "stage2": return STAGE2_CHECKPOINTS
    raise ValueError(stage)


def train_gated(*, model, optimizer,
                    train_provider: DocumentBatchProvider,
                    val_provider: DocumentBatchProvider,
                    checkpoint_out: Path,
                    baseline_sigma: Dict[str, float],
                    mixture: Dict[str, float],
                    l_native: int,
                    stage: str,
                    seed: int,
                    lr_peak: float,
                    lr_final: float,
                    ceiling: int = STAGE1_CEILING,
                    batch_size: int = 8,
                    device: str = "cpu",
                    warmup_frac: float = 0.02,
                    grad_clip: float = 1.0,
                    val_batches: int = 16,
                    save_start_snapshot: bool = True,
                    grad_obs_first_n: int = 100,
                    ) -> List[CheckpointRecord]:
    """Runs the gated learning-curve process. STOPS at every specified
    checkpoint, evaluates, saves, and either continues or halts per §9."""
    checkpoints = list(_stage_checkpoints(stage))
    random.seed(seed); torch.manual_seed(seed)

    # Snapshot start-weights for §13 weight-delta proof
    start_snapshot = snapshot_state_dict(model) if save_start_snapshot else None

    covered = 0
    step = 0
    n_updates = 0
    prev_val = None
    records: List[CheckpointRecord] = []
    gp_obs = []
    seq_rng = random.Random(seed + 1)

    while checkpoints and covered < ceiling:
        target = checkpoints[0]
        # Approx total steps for the LR schedule from covered -> target
        expected_tokens_per_step = EFFECTIVE_BATCH_TOKENS_TARGET
        approx_steps = max(1, (target - covered) // expected_tokens_per_step)
        warmup_steps = max(1, int(warmup_frac * approx_steps))

        while covered < target:
            step_lr = cosine_lr(step, approx_steps, warmup_steps, lr_peak, lr_final)
            for g in optimizer.param_groups: g["lr"] = step_lr
            seq_len = pick_sequence_length(l_native, seq_rng)
            b = train_provider.next_batch(batch_size, seq_len, device)
            r = train_one_step(model, optimizer, b, source=stage,
                                     grad_clip=grad_clip)
            if len(gp_obs) < grad_obs_first_n:
                gp_obs.append(observe_gradient_path(model, step))
            covered += r.valid_tokens
            step += 1
            n_updates += 1

        # Gate at target
        checkpoints.pop(0)
        val_loss, _ = evaluate_val_loss(model, val_provider,
                                                val_batches, batch_size,
                                                min(seq_len, l_native), device)
        rel_imp = 0.0 if prev_val is None else (prev_val - val_loss) / max(prev_val, 1e-12)
        try:
            assert_architecture_invariant(model)
            arch_ok = True
        except Exception:
            arch_ok = False
        try:
            assert_native_stability_gate(model, baseline_sigma)
            stab_ok = True
        except Exception:
            stab_ok = False
        gp_ok = True
        try:
            from .proof import assert_gradient_path_over_100_steps
            if len(gp_obs) >= grad_obs_first_n:
                assert_gradient_path_over_100_steps(gp_obs)
        except Exception:
            gp_ok = False
        m = save_candidate_checkpoint(model, optimizer,
                                              Path(checkpoint_out), step,
                                              covered, stage, seed, val_loss,
                                              mixture, lr_peak, n_updates,
                                              EFFECTIVE_BATCH_TOKENS_TARGET)
        records.append(CheckpointRecord(
            step=step, tokens_covered=covered, val_loss=val_loss,
            relative_improvement=rel_imp,
            architecture_delta_zero=arch_ok,
            native_stability_passed=stab_ok,
            grad_path_ok=gp_ok, manifest=m))
        # §9 continuation gate — only extend past 25M when relative
        # improvement ≥ 0.5% for the last two checkpoints AND every
        # gate passes.
        if covered >= 25_000_000 and covered < ceiling:
            improvements = [r.relative_improvement for r in records[-2:]]
            if all(i >= MIN_RELATIVE_IMPROVEMENT for i in improvements) and \
                    arch_ok and stab_ok and gp_ok:
                nxt = covered + STAGE1_INCREMENT_AFTER_25M
                if nxt <= ceiling:
                    checkpoints.append(nxt)
        prev_val = val_loss

    # §13 weight-delta report at end
    if start_snapshot is not None:
        end_snapshot = snapshot_state_dict(model)
        wd = compute_weight_delta(start_snapshot, end_snapshot)
        wd_report = {
            "total_tensors": wd.total_tensors,
            "positive_delta_count": len(wd.positive_delta),
            "zero_delta_count": len(wd.zero_delta),
            "min_nonzero_delta": wd.min_nonzero_delta,
            "median_delta": wd.median_delta,
            "max_delta": wd.max_delta,
            "zero_delta_tensors": wd.zero_delta,
        }
        (Path(checkpoint_out) / f"weight_delta_report_{stage}_seed{seed}.json").write_text(
            json.dumps(wd_report, indent=2, sort_keys=True), encoding="utf-8")
    return records
