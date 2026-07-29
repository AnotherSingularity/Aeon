"""aeon/job/worker.py — training worker entry point for Aeon.exe --worker JOB.

Runs the certified training loop under scripts/train.py's discipline but
inside a well-defined job dir with the safe-stop / safe-checkpoint protocol
from §W3. This module IS heavy (imports torch); aeon.entry keeps it lazy so
the launcher does not pull it in for --version or --verify-installation.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

from aeon.job.manager import (
    Job, JobStatus, load_job, mark_status, is_stop_requested,
    is_emergency_terminate_requested,
)
from aeon.job.identity import worker_identity
from aeon.job.lock import SingleInstanceLock, LockAcquireError


def _write_worker_identity(job: Job) -> None:
    ident = worker_identity()
    Path(job.worker_identity_path).write_text(
        json.dumps(ident.to_dict(), sort_keys=True), encoding="utf-8")
    Path(job.worker_pid_path).write_text(str(os.getpid()), encoding="utf-8")


def run_worker(job_file: str) -> int:
    """Load the job, acquire single-instance lock on the checkpoint dir, run
    training with a safe-stop hook, save a final atomic authenticated ckpt on
    stop, write result.json, exit."""
    job = load_job(job_file)
    if job is None:
        print(f"aeon: worker: job file not found or malformed: {job_file}", file=sys.stderr)
        return 5     # EXIT_JOB_NOT_FOUND

    # Single-instance: no two workers on the same checkpoint chain.
    lock_path = os.path.join(job.checkpoint_dir, ".aeon.lock")
    lock = SingleInstanceLock(lock_path)
    try:
        lock.acquire(timeout_s=0.5)
    except LockAcquireError as e:
        mark_status(job, JobStatus.FAILED,
                     note=f"another worker holds the checkpoint lock: {e}")
        return 8     # EXIT_WORKER_FAILED

    _write_worker_identity(job)
    mark_status(job, JobStatus.STARTING, note="worker acquired lock; loading config")

    try:
        _run_training_loop(job)
    except SystemExit as se:
        return int(se.code or 0)
    except Exception as e:
        mark_status(job, JobStatus.FAILED, note=f"crashed: {e!r}")
        _write_result(job, {"ok": False, "error": repr(e)})
        return 8
    finally:
        lock.release()

    return 0


def _write_result(job: Job, payload: dict) -> None:
    from aeon.job.manager import _atomic_write_json
    payload.setdefault("ts", time.time())
    _atomic_write_json(job.result_json_path, payload)


def _run_training_loop(job: Job) -> None:
    """Thin wrapper on scripts/train.py behaviour with a safe-stop hook.

    Imports torch here — worker only. Reuses the certified training helpers so
    every inherited invariant (fp32 recursion, γ recast, atomic checkpoint,
    strict resume, observability) continues to hold.
    """
    import yaml
    import torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    from aeon.checkpoint import (atomic_save, strict_load, build_metadata,
                                  latest_checkpoint as latest_ckpt)
    from aeon.observability import (Observer, parameter_accounting,
                                     optimizer_bytes_estimate, state_bytes,
                                     static_op_estimates,
                                     checkpoint_size_estimate, resident_mb)

    cfg = yaml.safe_load(open(job.config_path, encoding="utf-8"))
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[mcfg.get("dtype", "bfloat16")]

    torch.manual_seed(tcfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tcfg_model = AeonTransformerConfig(**mcfg.get("transformer", {}))
    model = HybridModel(
        h_rec=mcfg["h_rec"], K=mcfg["K"], transformer_config=tcfg_model,
        substrate=mcfg.get("substrate"), margin_h=mcfg["margin_h"],
        margin_c=mcfg["margin_c"], freeze_backbone=mcfg.get("freeze_backbone", False),
        use_embedding_input=mcfg.get("use_embedding_input", True), dtype=dtype,
    ).to(device)
    model.to(dtype=dtype)
    model.recursion.float()
    model.transformer.gamma.data = model.transformer.gamma.data.float()
    fb = getattr(model.substrate, "feedback", None)
    if fb is not None and isinstance(fb.gate_alpha, torch.nn.Parameter):
        fb.gate_alpha.data = fb.gate_alpha.data.float()
        fb.gate_threshold.data = fb.gate_threshold.data.float()

    params = model.trainable_parameters()
    opt = torch.optim.AdamW(params, lr=tcfg["lr"],
                             weight_decay=tcfg.get("weight_decay", 0.0))

    obs = Observer(out_dir=job.metrics_dir,
                    sample_every=int(tcfg.get("sample_every", 512)),
                    enabled=bool(tcfg.get("observability", True)))
    obs.emit_static("parameter_accounting", parameter_accounting(model))
    obs.emit_static("static_accounting", {
        "optimizer_bytes_estimate": optimizer_bytes_estimate(model, "adamw"),
        **state_bytes(model),
        "static_op_estimates": static_op_estimates(model, dcfg["seq_len"], mcfg["K"]),
        "checkpoint_bytes_estimate": checkpoint_size_estimate(model),
    })

    # Resume if a valid checkpoint exists
    start_step, data_position = 0, 0
    if tcfg.get("resume"):
        ck = latest_ckpt(job.checkpoint_dir)
        if ck:
            blob = strict_load(ck, expected_model_config=mcfg)
            model.load_state_dict(blob["model"])
            opt.load_state_dict(blob["optim"])
            start_step = int(blob["metadata"]["step"])
            data_position = int(blob["metadata"].get("data_position", 0))
            rng = blob.get("rng") or {}
            if "torch_cpu" in rng:
                torch.random.set_rng_state(rng["torch_cpu"])
            mark_status(job, JobStatus.STARTING,
                         note=f"resumed from {os.path.basename(ck)} step={start_step}")

    # Batch iterator — synthetic in this worker until tokenizer+corpus are set.
    B, T = tcfg["batch_size"], dcfg["seq_len"]
    g = torch.Generator(device=device).manual_seed(tcfg["seed"])
    def next_batch():
        return torch.randint(0, tcfg_model.vocab_size, (B, T),
                              generator=g, device=device)

    model.train()
    mark_status(job, JobStatus.RUNNING, note="training loop started",
                 step=start_step)

    step = start_step
    while step < tcfg["max_steps"]:
        # ---- safe-stop check at CHECKPOINT BOUNDARIES only ----------------
        if is_stop_requested(job) and step > start_step:
            mark_status(job, JobStatus.STOP_REQUESTED,
                         note="stop request observed at checkpoint boundary")
            _save_checkpoint(job, step, model, opt, mcfg, tcfg, dcfg,
                              data_position, obs)
            mark_status(job, JobStatus.STOPPED, note="safe stop complete", step=step)
            _write_result(job, {"ok": True, "stopped_at_step": step,
                                 "reason": "safe_stop_requested"})
            return
        if is_emergency_terminate_requested(job):
            mark_status(job, JobStatus.FAILED,
                         note="emergency terminate requested — no final checkpoint")
            _write_result(job, {"ok": False, "reason": "emergency_terminate"})
            return

        ids = next_batch()
        out = model(input_ids=ids, labels=ids)
        loss = out.loss
        beta = float(tcfg.get("aux_gate_penalty", 0.0))
        if beta and out.gate_mean is not None:
            loss = loss + beta * out.gate_mean
        opt.zero_grad(set_to_none=True); loss.backward()
        if tcfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
        opt.step()
        step += 1
        data_position += B * T

        if step % int(tcfg.get("log_every", 20)) == 0:
            a = model.audit()
            obs.emit_always_on(
                step=step, loss=float(out.loss.item()), lr=opt.param_groups[0]["lr"],
                step_time_s=0.0, tokens_per_s_raw=0.0, useful_tokens_per_s=0.0,
                seq_len=T, resident_mb=resident_mb(),
                certificate_holds=bool(a["holds"]),
                sigma_h=float(a["sigma_Wh"]), sigma_c=float(a["sigma_Wc"]),
                gamma=float(a["gamma"]),
            )
            mark_status(job, JobStatus.RUNNING, note="step log", step=step,
                         loss=float(out.loss.item()),
                         holds=bool(a["holds"]))
        if step % int(tcfg.get("ckpt_every", 1000)) == 0:
            _save_checkpoint(job, step, model, opt, mcfg, tcfg, dcfg,
                              data_position, obs)

    # Final checkpoint on clean completion
    _save_checkpoint(job, step, model, opt, mcfg, tcfg, dcfg, data_position, obs)
    mark_status(job, JobStatus.STOPPED, note="max_steps reached", step=step)
    _write_result(job, {"ok": True, "final_step": step, "reason": "max_steps"})


def _save_checkpoint(job: Job, step: int, model, opt, mcfg, tcfg, dcfg,
                      data_position: int, obs) -> None:
    from aeon.checkpoint import atomic_save, build_metadata
    mark_status(job, JobStatus.CHECKPOINTING, note="saving atomic checkpoint",
                 step=step)
    path = os.path.join(job.checkpoint_dir, f"ckpt_{step}.pt")
    md = build_metadata(step=step, model_cfg=mcfg, train_cfg=tcfg, data_cfg=dcfg,
                         tokenizer_id=job.tokenizer_path,
                         corpus_id=job.corpus_path,
                         data_position=int(data_position),
                         instrumentation_cfg={
                             "sample_every": int(tcfg.get("sample_every", 512)),
                             "enabled": bool(tcfg.get("observability", True))})
    atomic_save(path, model=model, optimizer=opt, metadata=md)
    mark_status(job, JobStatus.RUNNING, note=f"checkpoint saved: {os.path.basename(path)}",
                 step=step)
