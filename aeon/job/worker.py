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


def _cpu_thread_limit_from(job: Job, tcfg: dict):
    """W10-9/A7: honor cpu_thread_limit from job.compute_policy or tcfg."""
    v = None
    cp = getattr(job, "compute_policy", None) or {}
    if isinstance(cp, dict):
        v = cp.get("cpu_thread_limit")
    if v is None:
        v = tcfg.get("cpu_thread_limit")
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _memory_ceiling_mb_from(job: Job, tcfg: dict):
    """W10-9/A7: honor memory_ceiling_gb; returned as MB for comparison
    against resident_mb() from the Observer."""
    v = None
    cp = getattr(job, "compute_policy", None) or {}
    if isinstance(cp, dict):
        v = cp.get("memory_ceiling_gb")
    if v is None:
        v = tcfg.get("memory_ceiling_gb")
    try:
        return int(float(v) * 1024) if v is not None else None
    except (TypeError, ValueError):
        return None


def _validation_interval_from(job: Job, tcfg: dict):
    """W10-9/A7: honor validation_interval configured by the launcher."""
    v = None
    cp = getattr(job, "checkpoint_policy", None) or {}
    if isinstance(cp, dict):
        v = cp.get("validation_interval")
    if v is None:
        v = tcfg.get("validation_interval")
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


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
    from aeon.checkpoint import build_metadata
    from aeon.protected_checkpoint import (
        protected_load,
        CheckpointAuthenticationError, AntiRollbackViolation,
    )
    from aeon.job.key_store import ensure_job_hmac_keyref
    from aeon.job.generation import (
        generation_save, latest_authorized_generation, discard_incomplete,
        Generation as _Gen,
    )
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

    # W10-9/A7: honor the launcher-configured cpu_thread_limit. The Job
    # dataclass copies this out of user_config.json into
    # job.compute_policy["cpu_thread_limit"]; the worker previously ignored
    # it and let torch use all cores, which competes with the desktop
    # shell. Applied before model construction so the fp32 recursion build
    # obeys the same limit.
    cpu_limit = _cpu_thread_limit_from(job, tcfg)
    if cpu_limit is not None and cpu_limit > 0:
        try:
            torch.set_num_threads(int(cpu_limit))
        except Exception:
            pass
    memory_ceiling_mb = _memory_ceiling_mb_from(job, tcfg)
    validation_interval = _validation_interval_from(job, tcfg)

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
    # W10-R/R8: tokenizer vocab and model vocab must agree BEFORE model
    # construction. The old silent rebind was a fail-safe (prevented
    # embedding OOB) but violated the directive's fail-closed principle
    # — a misconfigured job needs to be visible to the operator, not
    # silently corrected. A caller who genuinely wants the tokenizer's
    # vocab to drive the model config declares it by leaving
    # transformer.vocab_size at 0 or omitting it entirely; in that
    # narrow case the rebind still applies.
    _tok_vocab = int(data_source.tokenizer.vocab_size)
    _cfg_vocab = int(tcfg_model.vocab_size)
    _cfg_uninit = ("vocab_size" not in mcfg.get("transformer", {})
                    or _cfg_vocab in (0, None))
    if _cfg_uninit:
        tcfg_model.vocab_size = _tok_vocab
    elif _tok_vocab != _cfg_vocab:
        mark_status(job, JobStatus.FAILED,
                     note=(f"data unavailable (tokenizer_vocab_mismatch): "
                           f"tokenizer vocab={_tok_vocab} != model config "
                           f"vocab={_cfg_vocab}"))
        _write_result(job, {"ok": False,
                             "reason": "data_unavailable",
                             "code": "tokenizer_vocab_mismatch",
                             "detail": (f"tokenizer.vocab_size={_tok_vocab} vs "
                                          f"config.transformer.vocab_size={_cfg_vocab}")})
        raise RuntimeError("data_unavailable:tokenizer_vocab_mismatch")
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

    # W10-4: sweep incomplete generation-*.tmp/ from any prior crash before
    # deciding whether a "start" refusal fires. An incomplete generation
    # cannot count as an active chain.
    discard_incomplete(job.checkpoint_dir)

    if intent == "start":
        # Fresh training. Refuse to overwrite an existing chain in the target
        # checkpoint_dir — that's the audit's "no accidental overwrite" gate.
        existing_ck = latest_authorized_generation(job.checkpoint_dir)
        if existing_ck is not None:
            mark_status(job, JobStatus.FAILED,
                         note=(f"start-new refused: checkpoint_dir already "
                               f"contains an authenticated generation "
                               f"({os.path.basename(existing_ck.path)}); pick Resume "
                               f"or use a fresh checkpoint_dir"))
            _write_result(job, {"ok": False, "reason": "start_new_refused_active_chain",
                                 "existing_generation": existing_ck.path})
            raise RuntimeError("start_new_refused_active_chain")

    elif intent in ("resume", "recover"):
        # Both intents load an authenticated checkpoint. The difference is
        # WHICH one and under WHAT authorization.
        if intent == "resume":
            if job.resume_from_checkpoint:
                ck = job.resume_from_checkpoint
            else:
                gen = latest_authorized_generation(job.checkpoint_dir)
                ck = gen.state_path if gen is not None else None
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
            if job.resume_from_checkpoint:
                ck = job.resume_from_checkpoint
            else:
                gen = latest_authorized_generation(job.checkpoint_dir)
                ck = gen.state_path if gen is not None else None
            if ck is None:
                mark_status(job, JobStatus.FAILED,
                             note="recover refused: no checkpoint to recover from")
                _write_result(job, {"ok": False, "reason": "recover_no_checkpoint"})
                raise RuntimeError("recover_no_checkpoint")

        # W10-R/R20: on Resume, bind the running release identity so a
        # cross-release load is refused. On Recovery, leave it None; the
        # RecoveryDecision is the explicit operator authorization for
        # the cross-release load.
        _expected_release = None
        if intent == "resume":
            from aeon.version import RELEASE_METADATA as _RM
            _expected_release = _RM.get("source_commit")
            if _expected_release == "unknown":
                _expected_release = None
        try:
            blob = protected_load(ck, keyref_mac=keyref,
                                   expected_model_config=mcfg,
                                   recovery_decision=recovery_decision,
                                   expected_release_identity=_expected_release)
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
        # W10-11: authorized rollback semantics — recovery to step N
        # must discard every generation with step > N that would
        # otherwise sit in the checkpoint dir ahead of the rollback
        # anchor. This is the semantic contract of RecoveryDecision:
        # the operator explicitly declares the newer generations
        # unfit. Without this discard, a subsequent Resume would pick
        # them back up. The tampered / corrupted state is preserved on
        # disk only as long as needed by this loop; the worker moves
        # them aside so future enumerators no longer surface them.
        if intent == "recover":
            _discard_generations_after(job.checkpoint_dir, start_step)
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
    # W10-9/A8: measure real step time so the emitted metrics stop being
    # zero placeholders. _step_perf tracks the total wall-clock spent
    # inside forward/backward/opt.step + tokens processed since the last
    # emission window; the log tick divides by the interval.
    from time import perf_counter as _pc
    _step_perf = {"t0": _pc(), "wallclock_s": 0.0, "tokens": 0}
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

        _step_t_before = _pc()
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
        _step_dt = _pc() - _step_t_before
        _step_perf["wallclock_s"] += _step_dt
        _step_perf["tokens"] += B * T
        step += 1
        data_position = position_after

        # W10-9/A7: honor memory_ceiling_gb — safe-stop at the next boundary
        # if resident memory has climbed above the launcher-configured limit.
        if memory_ceiling_mb is not None and resident_mb() > memory_ceiling_mb:
            mark_status(job, JobStatus.STOP_REQUESTED,
                         note=(f"memory ceiling exceeded: "
                               f"{resident_mb():.0f} MB > {memory_ceiling_mb} MB — "
                               "safe-stopping at boundary"),
                         step=step)
            _save_checkpoint(job, step, model, opt, mcfg, tcfg, dcfg,
                              data_position, obs,
                              tokenizer_id=data_source.tokenizer_id,
                              corpus_id=data_source.corpus_id,
                              keyref=keyref)
            mark_status(job, JobStatus.STOPPED,
                         note="safe stop on memory ceiling", step=step)
            _write_result(job, {"ok": True, "stopped_at_step": step,
                                 "reason": "memory_ceiling_exceeded",
                                 "resident_mb": resident_mb(),
                                 "memory_ceiling_mb": memory_ceiling_mb,
                                 "data_position": int(data_position),
                                 "tokenizer_id": data_source.tokenizer_id,
                                 "corpus_id": data_source.corpus_id})
            return

        if step % int(tcfg.get("log_every", 20)) == 0:
            a = model.audit()
            # W10-9/A8: real metrics. `step_time_s` is the average
            # wall-clock across the interval; tokens/s is real batches
            # processed / wall-clock — no more zero placeholders. Useful
            # tokens/s equals raw tokens/s because W10-1 made the token
            # stream real (attention masks cover all positions in the
            # non-padded corpus batches).
            _elapsed = max(_step_perf["wallclock_s"], 1e-9)
            _interval = max(int(tcfg.get("log_every", 20)), 1)
            step_time_s = _elapsed / _interval
            tokens_per_s = _step_perf["tokens"] / _elapsed
            _step_perf["wallclock_s"] = 0.0
            _step_perf["tokens"] = 0
            obs.emit_always_on(
                step=step, loss=float(out.loss.item()), lr=opt.param_groups[0]["lr"],
                step_time_s=step_time_s,
                tokens_per_s_raw=tokens_per_s,
                useful_tokens_per_s=tokens_per_s,
                seq_len=T, resident_mb=resident_mb(),
                certificate_holds=bool(a["holds"]),
                sigma_h=float(a["sigma_Wh"]), sigma_c=float(a["sigma_Wc"]),
                gamma=float(a["gamma"]),
            )
            mark_status(job, JobStatus.RUNNING, note="step log", step=step,
                         loss=float(out.loss.item()),
                         holds=bool(a["holds"]))

        # W10-9/A7 + W10-R/R6: honor validation_interval. Every N steps,
        # switch to eval mode, run a fresh forward pass under
        # torch.no_grad(), and restore train mode. Records an evidence
        # line the desktop Validate button can display. This validation
        # is separated from the in-flight training `out` so no training
        # RNG state is consulted and no optimizer step is possible.
        if (validation_interval is not None and validation_interval > 0
                and step % validation_interval == 0):
            _run_periodic_validation(job, model, batch, step)
        if step % int(tcfg.get("ckpt_every", 1000)) == 0:
            _save_checkpoint(job, step, model, opt, mcfg, tcfg, dcfg,
                              data_position, obs,
                              tokenizer_id=data_source.tokenizer_id,
                              corpus_id=data_source.corpus_id,
                              keyref=keyref)

    # Final checkpoint on clean completion. Guard against a double-save
    # when max_steps landed exactly on a ckpt_every boundary (the periodic
    # tick already wrote generation-<step>).
    latest = latest_authorized_generation(job.checkpoint_dir)
    if latest is None or int(latest.step) < step:
        _save_checkpoint(job, step, model, opt, mcfg, tcfg, dcfg, data_position, obs,
                          tokenizer_id=data_source.tokenizer_id,
                          corpus_id=data_source.corpus_id,
                          keyref=keyref)
    mark_status(job, JobStatus.STOPPED, note="max_steps reached", step=step)
    _write_result(job, {"ok": True, "final_step": step, "reason": "max_steps",
                         "data_position": int(data_position),
                         "tokenizer_id": data_source.tokenizer_id,
                         "corpus_id": data_source.corpus_id})


def _discard_generations_after(checkpoint_dir: str, keep_step: int) -> None:
    """W10-11: rollback anchor cleanup. Move every ``generation-<N>/``
    directory with N > keep_step aside so future enumerators no longer
    surface them. Uses rename (not delete) into a ``.discarded/`` sibling
    so evidence is preserved for post-mortem — the audit trail still
    contains the original bytes.
    """
    if not os.path.isdir(checkpoint_dir):
        return
    from aeon.job.generation import parse_generation_dir
    graveyard = os.path.join(checkpoint_dir, ".discarded")
    os.makedirs(graveyard, exist_ok=True)
    for name in sorted(os.listdir(checkpoint_dir)):
        parsed = parse_generation_dir(name)
        if parsed is None:
            continue
        gen_step, is_tmp = parsed
        if is_tmp:
            continue
        if gen_step > keep_step:
            src = os.path.join(checkpoint_dir, name)
            dst = os.path.join(graveyard, name)
            # Keep evidence — if a collision exists (unlikely), append a
            # numeric suffix rather than overwriting.
            candidate = dst
            suffix = 1
            while os.path.exists(candidate):
                candidate = f"{dst}.{suffix}"
                suffix += 1
            os.rename(src, candidate)
    # Reset latest-authorized pointer to keep_step so a subsequent
    # enumerator/loader agrees with the rollback.
    pointer = os.path.join(checkpoint_dir, "latest-authorized.txt")
    keeper = os.path.join(checkpoint_dir,
                             f"generation-{int(keep_step):08d}")
    if os.path.isdir(keeper):
        try:
            with open(pointer, "w", encoding="utf-8") as fh:
                fh.write(os.path.basename(keeper) + "\n")
        except Exception:
            pass


def _run_periodic_validation(job: Job, model, batch, step: int) -> None:
    """W10-9/A7 + W10-R/R6: periodic in-worker validation.

    Runs a fresh forward pass under torch.no_grad() with the model
    switched to eval mode; restores training mode before returning.
    Never mutates the optimizer, never advances the corpus cursor,
    never touches the training RNG. Records a JSONL evidence line
    the launcher's Validate button can surface.

    ``batch`` is the training batch that just completed; the
    validation re-runs the forward pass on the same tokens under
    eval() so the certificate audit reflects the model's post-step
    state and the NaN sweep sees eval-mode logits rather than the
    train-mode ones. A separate held-out validation partition is a
    larger architectural change tracked as a W10-R follow-on; the
    directive's minimum-correct requirement (eval mode, no_grad, no
    training RNG mutation, no optimizer step, own audit log) is met.
    """
    import math
    import torch
    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            out = model(input_ids=batch["input_ids"],
                          attention_mask=batch["attention_mask"],
                          labels=batch["labels"])
            a = model.audit()
            nan = False
            for t in (out.loss, getattr(out, "logits", None)):
                if t is None:
                    continue
                if not torch.isfinite(t).all():
                    nan = True
                    break
            rec = {
                "ts": time.time(),
                "kind": "periodic_validation",
                "step": int(step),
                "eval_mode": True,
                "no_grad": True,
                "certificate_holds": bool(a["holds"]),
                "sigma_h": float(a["sigma_Wh"]),
                "sigma_c": float(a["sigma_Wc"]),
                "gamma": float(a["gamma"]),
                "loss_finite": bool(math.isfinite(float(out.loss.item()))),
                "logits_have_nan_or_inf": bool(nan),
            }
        path = os.path.join(job.audit_dir, "worker_validation.jsonl")
        os.makedirs(job.audit_dir, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
    except Exception:
        pass
    finally:
        if was_training:
            model.train()


def _save_checkpoint(job: Job, step: int, model, opt, mcfg, tcfg, dcfg,
                      data_position: int, obs,
                      *, tokenizer_id: str, corpus_id: str, keyref) -> None:
    """Save one PROTECTED, ATOMIC generation (W10-4). The full envelope
    (payload + meta.json + sha256 + COMPLETE marker) is written into a
    ``generation-<step>.tmp/`` directory and only promoted to
    ``generation-<step>/`` after every component is verified round-trip.
    A crash between the payload write and the metadata write leaves a
    .tmp directory that no loader ever selects; the previous generation
    remains authoritatively selectable via ``latest-authorized.txt``.
    """
    from aeon.checkpoint import build_metadata
    from aeon.job.generation import generation_save

    mark_status(job, JobStatus.CHECKPOINTING,
                 note=f"saving protected generation-{step}", step=step)
    md = build_metadata(step=step, model_cfg=mcfg, train_cfg=tcfg, data_cfg=dcfg,
                         tokenizer_id=tokenizer_id,
                         corpus_id=corpus_id,
                         data_position=int(data_position),
                         instrumentation_cfg={
                             "sample_every": int(tcfg.get("sample_every", 512)),
                             "enabled": bool(tcfg.get("observability", True))})
    gen = generation_save(job.checkpoint_dir, step,
                           model=model, optimizer=opt, metadata=md,
                           keyref=keyref, authorized_step=int(step))
    mark_status(job, JobStatus.RUNNING,
                 note=f"protected generation saved: {os.path.basename(gen.path)}",
                 step=step)
