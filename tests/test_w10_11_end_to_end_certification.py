"""W10-11 — End-to-end certification against a bounded English fixture.

The final W10 tranche's job is to demonstrate that all preceding
corrective tranches (W10-1..W10-10) compose end-to-end. It runs the
Windows worker over a real English tokenizer + real English JSONL
corpus, exercises the full lifecycle, and asserts that:

    * Start produces a protected authenticated checkpoint (W10-2/A4,
      W10-4/A16).
    * Metrics include real perf_counter timings and non-zero throughput
      (W10-9/A8).
    * Periodic worker validation records evidence (W10-9/A7).
    * Resume from the authenticated checkpoint continues training
      forward past the resume step (W10-3/A6).
    * Corrupting the payload of the newest generation causes the
      verifier to reject that checkpoint on next Resume (W10-2, W10-6).
    * A Recovery flow with a BuildableRecoveryDecision + hmac keyref
      round-trip loads the PREVIOUS authenticated generation and the
      worker continues from that anchor (W10-3, W10-4).

The fixture is intentionally tiny (small vocab, tiny corpus, ~6 steps
per phase, 64-dim hidden) so the whole cycle runs in seconds. It is
NOT a scientific evaluation — it is the certification that the
lifecycle machinery does not silently regress.
"""
import json
import os
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def _train_fixture_tokenizer(dst_prefix: str):
    import sentencepiece as spm
    raw = Path(dst_prefix).parent / "raw.txt"
    sentences = ["the quick brown fox jumps over the lazy dog",
                 "a small red car parked on the street",
                 "she opened the door and walked outside",
                 "he laughed and closed the book slowly",
                 "birds sing at dawn on the old oak tree",
                 "the ship sailed across the deep blue sea",
                 "children played in the sunny green park",
                 "we watched the clouds drift over the hill"]
    raw.write_text("\n".join(sentences * 60))
    spm.SentencePieceTrainer.train(
        input=str(raw), model_prefix=dst_prefix, vocab_size=200,
        character_coverage=0.9995, model_type="bpe",
        bos_id=1, eos_id=2, pad_id=0, unk_id=3)
    return dst_prefix + ".model", "\n".join(sentences * 4)


def _corpus_jsonl(path: Path, text_body: str, n: int = 8):
    """Write a corpus large enough that the tiny 6-step training loop
    never exhausts it and terminates early."""
    path.write_text("\n".join(json.dumps({"text": text_body})
                                 for _ in range(n)) + "\n")


def _cfg(vocab_size: int) -> dict:
    return {
        "model": {
            "h_rec": 64, "K": 16, "margin_h": 0.02, "margin_c": 0.02,
            "dtype": "float32",
            "transformer": {"hidden_size": 64, "num_hidden_layers": 1,
                              "num_attention_heads": 2,
                              "num_key_value_heads": 2, "head_dim": 32,
                              "intermediate_size": 128,
                              "vocab_size": vocab_size,
                              "max_position_embeddings": 32}
        },
        "data": {"seq_len": 32},
        "train": {"seed": 1, "lr": 0.001, "batch_size": 2,
                    "max_steps": 6, "log_every": 2, "ckpt_every": 2,
                    "sample_every": 1000, "observability": True,
                    "aux_gate_penalty": 0.0},
    }


def _create_job(d: Path, cfg_path: Path, tok_model: str, corpus_dir: Path,
                  *, intent: str = "start", resume_from: str = None,
                  recovery_decision_path: str = None):
    from aeon.job.manager import create_job
    return create_job(
        config_path=str(cfg_path),
        tokenizer_path=tok_model,
        corpus_path=str(corpus_dir),
        checkpoint_dir=str(d / "ck"),
        metrics_dir=str(d / "metrics"),
        audit_dir=str(d / "audit"),
        checkpoint_policy={"interval": 2, "validation_interval": 2},
        compute_policy={"cpu_thread_limit": 1},
        intent=intent,
        resume_from_checkpoint=resume_from,
        recovery_decision_path=recovery_decision_path,
    )


def _first_authenticated_metadata(candidates):
    for c in candidates:
        if getattr(c, "authenticated", False):
            return c
    return None


# ---------------------------------------------------------------------------
def test_end_to_end_start_resume_corrupt_recover():
    try:
        import sentencepiece as spm  # noqa
        import torch  # noqa
        import yaml
    except Exception:
        return  # skip if deps are absent
    from aeon.job.worker import run_worker
    from aeon.job.key_store import ensure_job_hmac_keyref
    from aeon.launcher.resume import (
        enumerate_checkpoints, latest_authenticated_checkpoint,
        BuildableRecoveryDecision,
    )

    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        # ---- English fixture --------------------------------------------
        tok_model, text_body = _train_fixture_tokenizer(str(d / "tok"))
        corpus_dir = d / "corpus"
        corpus_dir.mkdir()
        _corpus_jsonl(corpus_dir / "shard0.jsonl", text_body)
        # Read vocab out to make config match
        from aeon.tokenizer import AeonTokenizer
        vocab_size = AeonTokenizer(tok_model).vocab_size
        cfg = _cfg(vocab_size)
        cfg_path = d / "cfg.yaml"
        cfg_path.write_text(yaml.safe_dump(cfg))

        # =============================================================
        # Phase 1: Start
        # =============================================================
        with mock.patch("aeon.job.manager.jobs_dir", return_value=d / "jobs"):
            j_start = _create_job(d, cfg_path, tok_model, corpus_dir,
                                     intent="start")
            rc = run_worker(j_start.job_json_path)
            assert rc == 0, (
                "W10-11: start worker failed rc=" + str(rc)
                + " result=" + Path(j_start.result_json_path).read_text())
        # Metrics must include at least one non-zero throughput line.
        metrics = list((d / "metrics").glob("*.jsonl"))
        assert metrics, "W10-11: no metrics.jsonl written"
        emitted = [json.loads(l) for l in metrics[0].read_text().splitlines()]
        emitted_always = [r for r in emitted if r.get("kind") == "always_on"]
        assert emitted_always, "W10-11: no always_on metric lines"
        assert any(float(r.get("tokens_per_s_raw", 0)) > 0
                     and float(r.get("step_time_s", 0)) > 0
                     for r in emitted_always), (
            "W10-11/A8: worker must emit non-zero throughput metrics")
        # Periodic validation evidence must exist.
        val_path = d / "audit" / "worker_validation.jsonl"
        assert val_path.exists(), (
            "W10-11/A7: worker_validation.jsonl must be present")
        assert val_path.read_text().strip(), (
            "W10-11/A7: worker_validation.jsonl must have content")
        # An authenticated checkpoint must be discoverable.
        keyref_start = ensure_job_hmac_keyref(j_start.job_dir,
                                                 allow_create=False)
        latest_after_start = latest_authenticated_checkpoint(
            str(d / "ck"), keyref_start)
        assert latest_after_start is not None, (
            "W10-11: Start must leave at least one authenticated checkpoint")
        step_after_start = latest_after_start.step

        # =============================================================
        # Phase 2: Resume
        # =============================================================
        with mock.patch("aeon.job.manager.jobs_dir", return_value=d / "jobs"):
            j_resume = _create_job(d, cfg_path, tok_model, corpus_dir,
                                      intent="resume")
            # Reuse the start job's HMAC key (same envelope must round-trip)
            import shutil
            shutil.copy(os.path.join(j_start.job_dir, "hmac.key"),
                          os.path.join(j_resume.job_dir, "hmac.key"))
            # Bump max_steps so resume genuinely progresses past the anchor.
            cfg2 = _cfg(vocab_size)
            cfg2["train"]["max_steps"] = step_after_start + 4
            cfg_path2 = d / "cfg2.yaml"
            cfg_path2.write_text(yaml.safe_dump(cfg2))
            # Rewrite the job's config_path to point at the extended config.
            j_resume.config_path = str(cfg_path2)
            from aeon.job.manager import _atomic_write_json
            _atomic_write_json(j_resume.job_json_path, j_resume.to_dict())
            rc = run_worker(j_resume.job_json_path)
            assert rc == 0, (
                "W10-11: resume worker failed rc=" + str(rc)
                + " result=" + Path(j_resume.result_json_path).read_text())
        latest_after_resume = latest_authenticated_checkpoint(
            str(d / "ck"), keyref_start)
        assert latest_after_resume.step > step_after_start, (
            "W10-11: resume must advance past the previous latest step")

        # =============================================================
        # Phase 3: Corrupt the newest generation, verify enumeration rejects
        # =============================================================
        # Tamper with the newest generation's state.pt by appending garbage.
        newest_path = latest_after_resume.path
        with open(newest_path, "ab") as fh:
            fh.write(b"\0\0\0TAMPERED\0\0\0")
        candidates_after_tamper = enumerate_checkpoints(
            str(d / "ck"), keyref_start)
        newest_after_tamper = [c for c in candidates_after_tamper
                                  if os.path.abspath(c.path)
                                     == os.path.abspath(newest_path)]
        assert newest_after_tamper, "tampered checkpoint disappeared from listing"
        assert newest_after_tamper[0].authenticated is False, (
            "W10-11/A4: tampered checkpoint must fail authentication")
        # And an earlier authenticated candidate must still exist.
        older_authenticated = [c for c in candidates_after_tamper
                                  if c.authenticated
                                  and os.path.abspath(c.path)
                                     != os.path.abspath(newest_path)]
        assert older_authenticated, (
            "W10-11: earlier authenticated candidate must remain after "
            "tampering the newest")
        recovery_target = older_authenticated[0]

        # =============================================================
        # Phase 4: Recovery (build RecoveryDecision + spawn intent=recover)
        # =============================================================
        current_state_identity = (
            f"sha256:{latest_after_resume.mac_algo}:step="
            f"{latest_after_resume.step}")
        brd = BuildableRecoveryDecision(
            candidate=recovery_target,
            reason="W10-11 end-to-end certification: tampered newest generation",
            operator_authorization_ref="test:desktop_operator",
            current_state_identity=current_state_identity)
        rd = brd.build()
        with mock.patch("aeon.job.manager.jobs_dir", return_value=d / "jobs"):
            # Bump max_steps so recovery genuinely progresses past the anchor.
            cfg3 = _cfg(vocab_size)
            cfg3["train"]["max_steps"] = recovery_target.step + 4
            cfg_path3 = d / "cfg3.yaml"
            cfg_path3.write_text(yaml.safe_dump(cfg3))
            # Pre-write the RecoveryDecision so create_job's non-empty
            # recovery_decision_path guard passes; the file lives under
            # a temp location because the job_dir isn't known yet.
            rd_tmp = d / "rd.json"
            with open(rd_tmp, "w", encoding="utf-8") as fh:
                json.dump(rd.__dict__, fh, sort_keys=True, indent=2)
            j_recover = _create_job(
                d, cfg_path3, tok_model, corpus_dir,
                intent="recover",
                resume_from=recovery_target.path,
                recovery_decision_path=str(rd_tmp))
            import shutil
            shutil.copy(os.path.join(j_start.job_dir, "hmac.key"),
                          os.path.join(j_recover.job_dir, "hmac.key"))
            rc = run_worker(j_recover.job_json_path)
            assert rc == 0, (
                "W10-11: recover worker failed rc=" + str(rc)
                + " result=" + Path(j_recover.result_json_path).read_text())
        latest_after_recover = latest_authenticated_checkpoint(
            str(d / "ck"), keyref_start)
        assert latest_after_recover.step > recovery_target.step, (
            "W10-11: recovery must advance past the rollback anchor")

        # W10-11: authorized rollback moves the tampered generation into
        # <ck>/.discarded/ so its bytes are preserved as evidence, then
        # the recovery worker continues past the anchor and writes fresh
        # (authenticated) generations. The primary enumeration must
        # contain only authenticated candidates; the .discarded/ tree
        # holds the tampered original.
        graveyard = d / "ck" / ".discarded"
        assert graveyard.is_dir(), (
            "W10-11: authorized rollback must preserve evidence under "
            ".discarded/")
        graveyard_names = os.listdir(graveyard)
        assert any(name.startswith("generation-") for name in graveyard_names), (
            "W10-11: tampered generation must be moved into .discarded/ "
            f"for evidence, got {graveyard_names}")
        # Every checkpoint remaining in the primary enumeration must
        # authenticate — no tampered bytes remain discoverable via
        # enumerate_checkpoints.
        candidates_final = enumerate_checkpoints(
            str(d / "ck"), keyref_start)
        for c in candidates_final:
            assert c.authenticated, (
                f"W10-11: primary enumeration must not contain "
                f"unauthenticated candidates after rollback; got {c}")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
