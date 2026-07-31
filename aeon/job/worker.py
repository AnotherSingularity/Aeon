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
    from aeon.checkpoint import build_metadata, latest_checkpoint as latest_ckpt
    from aeon.protected_checkpoint import (
        protected_load,
        CheckpointAuthenticationError, AntiRollbackViolation,
    )
    from aeon.job.key_store import ensure_job_hmac_keyref
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

    # W10-1: real tokenizer + corpus is the ONLY production data path.
    # Fail closed BEFORE model construction so we never touch torch on an
    # invalid data configuration. DataSourceError is re-raised into the
    # outer except-Exception block, which writes result.json and marks
    # the job FAILED with a structured `data_unavailable` reason.
    from aeon.job.data_source import build_data_source, DataSourceError
    try:
        data_source = build_data_source(job, tcfg, dcfg)
    except DataSourceError as e:
        mark_status(job, JobStatus.FAILED,
                     note=f"data unavailable ({e.reason}): {e.detail}"[:400])
        _write_result(job, {"ok": False, "reason": "data_unavailable",
                             "code": e.reason, "detail": e.detail})
        # Bubble a plain RuntimeError so run_worker returns rc=8; the
        # structured code is already in result.json.
        raise RuntimeError(f"data_unavailable:{e.reason}") from e

    tcfg_model = AeonTransformerConfig(**mcfg.get("transformer", {}))
    # W10-1: bind the transformer's vocab_size to the loaded tokenizer's
    # vocab (matches scripts/train.py::main). Random-init cannot happen
    # against a vocab that does not match the tokenizer that produces the
    # ids.
    if data_source.tokenizer.vocab_size != tcfg_model.vocab_size:
        tcfg_model.vocab_size = data_source.tokenizer.vocab_size
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

    # W10-2: per-job HMAC key backs the protected checkpoint envelope. Created
    # on first save, reused across restarts, kept out of the runtime manifest.
    keyref = ensure_job_hmac_keyref(job.job_dir, allow_create=True)

    # W10-3: distinct Start / Resume / Recovery intents. The `intent` field on
    # the Job dataclass is the ONE source of truth. tcfg["resume"] is no
    # longer consulted — a config-file toggle could not distinguish the three
    # GUI paths.
    intent = getattr(job, "intent", "start")
    start_step, data_position = 0, 0

    if intent == "start":
        # Fresh training. Refuse to overwrite an existing chain in the target
        # checkpoint_dir — that's the audit's "no accidental overwrite" gate.
        existing_ck = latest_ckpt(job.checkpoint_dir)
        if existing_ck is not None:
            mark_status(job, JobStatus.FAILED,
                         note=(f"start-new refused: checkpoint_dir already "
                               f"contains an authenticated chain "
                               f"({os.path.basename(existing_ck)}); pick Resume "
                               f"or use a fresh checkpoint_dir"))
            _write_result(job, {"ok": False, "reason": "start_new_refused_active_chain",
                                 "existing_checkpoint": existing_ck})
            raise RuntimeError("start_new_refused_active_chain")

    elif intent in ("resume", "recover"):
        # Both intents load an authenticated checkpoint. The difference is
        # WHICH one and under WHAT authorization.
        if intent == "resume":
            ck = job.resume_from_checkpoint or latest_ckpt(job.checkpoint_dir)
            if ck is None:
                mark_status(job, JobStatus.FAILED,
                             note="resume refused: no eligible authenticated checkpoint")
                _write_result(job, {"ok": False, "reason": "resume_no_eligible_checkpoint"})
                raise RuntimeError("resume_no_eligible_checkpoint")
            recovery_decision = None
        else:  # recover
            from aeon.protected_checkpoint import RecoveryDecision as _RD
            rd_path = job.recovery_decision_path
            if not rd_path or not os.path.exists(rd_path):
                mark_status(job, JobStatus.FAILED,
                             note=f"recover refused: recovery_decision_path missing: {rd_path!r}")
                _write_result(job, {"ok": False, "reason": "recovery_decision_missing"})
                raise RuntimeError("recovery_decision_missing")
            try:
                rd_payload = json.loads(open(rd_path, encoding="utf-8").read())
                recovery_decision = _RD(**rd_payload)
            except Exception as e:
                mark_status(job, JobStatus.FAILED,
                             note=f"recover refused: malformed RecoveryDecision: {e}")
                _write_result(job, {"ok": False, "reason": "recovery_decision_malformed",
                                     "detail": str(e)})
                raise
            ck = job.resume_from_checkpoint or latest_ckpt(job.checkpoint_dir)
            if ck is None:
                mark_status(job, JobStatus.FAILED,
                             note="recover refused: no checkpoint to recover from")
                _write_result(job, {"ok": False, "reason": "recover_no_checkpoint"})
                raise RuntimeError("recover_no_checkpoint")

        try:
            blob = protected_load(ck, keyref_mac=keyref,
                                   expected_model_config=mcfg,
                                   recovery_decision=recovery_decision)
        except CheckpointAuthenticationError as e:
            mark_status(job, JobStatus.FAILED,
                         note=f"checkpoint authentication failed: {e}"[:400])
            _write_result(job, {"ok": False,
                                 "reason": "checkpoint_authentication_failed",
                                 "detail": str(e)})
            raise
        except AntiRollbackViolation as e:
            mark_status(job, JobStatus.FAILED,
                         note=f"anti-rollback: {e}"[:400])
            _write_result(job, {"ok": False,
                                 "reason": "anti_rollback_violation",
                                 "detail": str(e)})
            raise
        model.load_state_dict(blob["model"])
        opt.load_state_dict(blob["optim"])
        inner = blob["envelope_metadata"]["inner_metadata"]
        start_step = int(inner["step"])
        data_position = int(inner.get("data_position", 0))
        rng = blob.get("rng") or {}
        if "torch_cpu" in rng:
            torch.random.set_rng_state(rng["torch_cpu"])
        mark_status(job, JobStatus.STARTING,
                     note=(f"{intent} (protected) from "
                           f"{os.path.basename(ck)} step={start_step} "
                           f"authorized_step="
                           f"{blob['envelope_metadata'].get('authorized_step')}"))
    else:
        raise ValueError(f"unknown intent: {intent!r}")

    # W10-1: real corpus batches. The synthetic torch.randint next_batch is
    # gone. batches is a generator of (batch_dict, position_after) pairs
    # from the fail-closed data source constructed above.
    B, T = data_source.batch_size, data_source.seq_len
    batches = data_source.iter_batches(device=device,
                                        start_position=int(data_position))

    model.train()
    mark_status(job, JobStatus.RUNNING,
                 note=(f"training loop started; corpus_id="
                       f"{data_source.corpus_id[:23]}… "
                       f"tokenizer_id={data_source.tokenizer_id[:23]}… "
                       f"records={data_source.records} "
                       f"total_tokens={data_source.total_tokens}"),
                 step=start_step)

    step = start_step
    for batch, position_after in batches:
        if step >= tcfg["max_steps"]:
            break
        # ---- safe-stop check at CHECKPOINT BOUNDARIES only ----------------
        if is_stop_requested(job) and step > start_step:
            mark_status(job, JobStatus.STOP_REQUESTED,
                         note="stop request observed at checkpoint boundary")
            _save_checkpoint(job, step, model, opt, mcfg, tcfg, dcfg,
                              data_position, obs,
                              tokenizer_id=data_source.tokenizer_id,
                              corpus_id=data_source.corpus_id,
                              keyref=keyref)
            mark_status(job, JobStatus.STOPPED, note="safe stop complete", step=step)
            _write_result(job, {"ok": True, "stopped_at_step": step,
                                 "reason": "safe_stop_requested",
                                 "data_position": int(data_position),
                                 "tokenizer_id": data_source.tokenizer_id,
                                 "corpus_id": data_source.corpus_id})
            return
        if is_emergency_terminate_requested(job):
            mark_status(job, JobStatus.FAILED,
                         note="emergency terminate requested — no final checkpoint")
            _write_result(job, {"ok": False, "reason": "emergency_terminate"})
            return

        out = model(input_ids=batch["input_ids"],
                     attention_mask=batch["attention_mask"],
                     labels=batch["labels"])
        loss = out.loss
        beta = float(tcfg.get("aux_gate_penalty", 0.0))
        if beta and out.gate_mean is not None:
            loss = loss + beta * out.gate_mean
        opt.zero_grad(set_to_none=True); loss.backward()
        if tcfg.get("grad_clip"):
            torch.nn.utils.clip_grad_norm_(params, tcfg["grad_clip"])
        opt.step()
        step += 1
        data_position = position_after

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
                              data_position, obs,
                              tokenizer_id=data_source.tokenizer_id,
                              corpus_id=data_source.corpus_id,
                              keyref=keyref)

    # Final checkpoint on clean completion
    _save_checkpoint(job, step, model, opt, mcfg, tcfg, dcfg, data_position, obs,
                      tokenizer_id=data_source.tokenizer_id,
                      corpus_id=data_source.corpus_id)
    mark_status(job, JobStatus.STOPPED, note="max_steps reached", step=step)
    _write_result(job, {"ok": True, "final_step": step, "reason": "max_steps",
                         "data_position": int(data_position),
                         "tokenizer_id": data_source.tokenizer_id,
                         "corpus_id": data_source.corpus_id})


def _save_checkpoint(job: Job, step: int, model, opt, mcfg, tcfg, dcfg,
                      data_position: int, obs,
                      *, tokenizer_id: str, corpus_id: str, keyref) -> None:
    """Save a PROTECTED checkpoint (W10-2). This calls
    ``aeon.protected_checkpoint.protected_save`` which:

      * writes payload + `.meta.json` + `.sha256` atomically,
      * computes an HMAC-SHA256 tag over (payload bytes ‖ metadata json)
        under the per-job key returned by ``ensure_job_hmac_keyref``,
      * records ``authorized_step`` so a subsequent resume enforces
        anti-rollback,
      * binds K=K_LOCKED and the E3 patch manifest into the envelope.

    The launcher's Safe Stop message and the Resume selector can now
    truthfully call these checkpoints authenticated — the flipping of
    audit findings A4 and A5 is the whole point of this tranche.
    """
    from aeon.checkpoint import build_metadata
    from aeon.protected_checkpoint import protected_save

    mark_status(job, JobStatus.CHECKPOINTING,
                 note="saving protected authenticated checkpoint", step=step)
    path = os.path.join(job.checkpoint_dir, f"ckpt_{step}.pt")
    md = build_metadata(step=step, model_cfg=mcfg, train_cfg=tcfg, data_cfg=dcfg,
                         tokenizer_id=tokenizer_id,
                         corpus_id=corpus_id,
                         data_position=int(data_position),
                         instrumentation_cfg={
                             "sample_every": int(tcfg.get("sample_every", 512)),
                             "enabled": bool(tcfg.get("observability", True))})
    protected_save(path, model=model, optimizer=opt, metadata=md,
                    keyref_mac=keyref, authorized_step=int(step))
    mark_status(job, JobStatus.RUNNING,
                 note=f"protected checkpoint saved: {os.path.basename(path)}",
                 step=step)
