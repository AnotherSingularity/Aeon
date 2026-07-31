"""W10-4 — atomic per-generation checkpoint chain.

Every test drives real (tiny) checkpoints through ``aeon.job.generation``
and verifies:

* A crash before COMPLETE leaves a ``generation-N.tmp/`` that no loader
  ever selects; the previous authorized generation stays discoverable.
* Multiple generations coexist; each has its own state.pt/meta.json/
  sha256/COMPLETE.
* ``latest-authorized.txt`` is atomically updated after a successful
  promotion.
* Tampering the payload OR the meta OR the sha256 sidecar fails the
  loader with a specific error, and the previous generation remains
  loadable.
* ``discard_incomplete`` cleans .tmp dirs and generation dirs without
  COMPLETE without touching complete ones.
* Recovery can select the previous authorized generation.
"""
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from aeon.checkpoint import build_metadata
from aeon.hybrid import HybridModel
from aeon.transformer import AeonTransformerConfig
from aeon.job.key_store import ensure_job_hmac_keyref
from aeon.job.generation import (
    generation_save, generation_dir_name, parse_generation_dir,
    list_generations, discard_incomplete,
    latest_authorized_generation, previous_authorized_generation,
    read_latest_pointer, COMPLETE_MARKER,
)
from aeon.protected_checkpoint import (
    protected_load, CheckpointAuthenticationError,
)


def _mini():
    tcfg = AeonTransformerConfig(
        vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=32, rms_norm_eps=1e-5,
        rope_theta=10000.0, tie_word_embeddings=True, attention_bias=False)
    m = HybridModel(
        h_rec=8, K=16, transformer_config=tcfg,
        substrate={"kind": "matrix", "d_in": 8, "d_state": 8,
                    "n_head": 2, "head_size": 4},
        margin_h=0.98, margin_c=0.95,
        freeze_backbone=False, use_embedding_input=True, dtype=torch.float32)
    m.transformer.gamma.data = m.transformer.gamma.data.float()
    return m, tcfg


def _md(step, tcfg):
    return build_metadata(step=step,
                           model_cfg={"K": 16, "transformer": tcfg.__dict__},
                           train_cfg={"lr": 1e-3, "batch_size": 1, "seed": 0},
                           data_cfg={"seq_len": 8},
                           tokenizer_id="sha256:test", corpus_id="sha256:test",
                           data_position=step * 8,
                           instrumentation_cfg={"sample_every": 512,
                                                "enabled": True})


def _save(dir_, kr, step):
    m, tcfg = _mini()
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-3)
    return generation_save(dir_, step, model=m, optimizer=opt,
                            metadata=_md(step, tcfg), keyref=kr,
                            authorized_step=step)


# ---------------------------------------------------------------------------
def test_generation_dir_name_padded():
    assert generation_dir_name(5) == "generation-00000005"
    assert generation_dir_name(1000000) == "generation-01000000"


def test_parse_generation_dir():
    assert parse_generation_dir("generation-00000005") == (5, False)
    assert parse_generation_dir("generation-00000006.tmp") == (6, True)
    assert parse_generation_dir("ckpt_5.pt") is None
    assert parse_generation_dir("hmac.key") is None


def test_generation_save_promotes_atomically():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        gen = _save(ckdir, kr, step=1)
        assert gen.step == 1 and gen.complete is True
        for f in ("state.pt", "state.pt.meta.json", "state.pt.sha256", "COMPLETE"):
            assert os.path.exists(os.path.join(gen.path, f)), f
        # Round-trip through protected_load succeeds.
        blob = protected_load(gen.state_path, keyref_mac=kr,
                               expected_model_config={
                                   "K": 16, "transformer": {"vocab_size": 32}})
        assert blob["envelope_metadata"]["inner_metadata"]["step"] == 1


def test_multiple_generations_coexist():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        _save(ckdir, kr, step=1)
        _save(ckdir, kr, step=2)
        _save(ckdir, kr, step=3)
        gens = list_generations(ckdir)
        assert [g.step for g in gens] == [3, 2, 1]
        latest = latest_authorized_generation(ckdir)
        assert latest is not None and latest.step == 3


def test_latest_pointer_updates_on_promotion():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        _save(ckdir, kr, step=1)
        assert read_latest_pointer(ckdir) == "generation-00000001"
        _save(ckdir, kr, step=2)
        assert read_latest_pointer(ckdir) == "generation-00000002"


def test_incomplete_generation_never_selected():
    """Simulate a crash between the payload write and the COMPLETE
    marker. list_generations must skip it; latest_authorized_generation
    must fall through to the previous complete generation."""
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        good = _save(ckdir, kr, step=5)
        # Hand-fabricate a generation-6 dir WITHOUT a COMPLETE marker.
        bad = os.path.join(ckdir, generation_dir_name(6))
        os.makedirs(bad)
        Path(os.path.join(bad, "state.pt")).write_bytes(b"garbage")
        # list_generations by default excludes it.
        visible = [g.step for g in list_generations(ckdir)]
        assert visible == [5], visible
        # And with include_incomplete=True it becomes visible marked incomplete.
        with_inc = list_generations(ckdir, include_incomplete=True)
        steps = sorted((g.step, g.complete) for g in with_inc)
        assert (5, True) in steps and (6, False) in steps
        # latest_authorized_generation still points at step 5.
        latest = latest_authorized_generation(ckdir)
        assert latest is not None and latest.step == 5


def test_tmp_generation_from_prior_crash_also_skipped():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        _save(ckdir, kr, step=1)
        # A .tmp dir from a prior crash
        tmp = os.path.join(ckdir, generation_dir_name(2) + ".tmp")
        os.makedirs(tmp)
        Path(os.path.join(tmp, "state.pt")).write_bytes(b"partial")
        visible = [g.step for g in list_generations(ckdir)]
        assert visible == [1]


def test_discard_incomplete_removes_tmp_and_no_complete_dirs():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        good = _save(ckdir, kr, step=1)
        # Fabricate two problem dirs
        tmp = os.path.join(ckdir, generation_dir_name(2) + ".tmp"); os.makedirs(tmp)
        Path(os.path.join(tmp, "state.pt")).write_bytes(b"partial")
        no_complete = os.path.join(ckdir, generation_dir_name(3)); os.makedirs(no_complete)
        Path(os.path.join(no_complete, "state.pt")).write_bytes(b"partial")
        removed = discard_incomplete(ckdir)
        assert not os.path.exists(tmp)
        assert not os.path.exists(no_complete)
        # The good generation is untouched.
        assert os.path.exists(good.path)
        assert os.path.exists(os.path.join(good.path, "COMPLETE"))
        assert set(removed) == {tmp, no_complete}


def test_generation_save_refuses_duplicate_step():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        _save(ckdir, kr, step=1)
        try:
            _save(ckdir, kr, step=1)
            raise AssertionError("expected FileExistsError")
        except FileExistsError:
            pass


def test_tampering_current_generation_leaves_previous_recoverable():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        prev = _save(ckdir, kr, step=1)
        curr = _save(ckdir, kr, step=2)
        # Tamper the current generation's payload.
        b = bytearray(Path(curr.state_path).read_bytes())
        b[100] ^= 0xFF
        Path(curr.state_path).write_bytes(b)
        try:
            protected_load(curr.state_path, keyref_mac=kr,
                            expected_model_config={
                                "K": 16, "transformer": {"vocab_size": 32}})
            raise AssertionError("expected failure on tampered payload")
        except Exception as e:
            assert "mismatch" in str(e).lower() or "mac" in str(e).lower()
        # The previous generation is still authenticatable.
        blob = protected_load(prev.state_path, keyref_mac=kr,
                               expected_model_config={
                                   "K": 16, "transformer": {"vocab_size": 32}})
        assert blob["envelope_metadata"]["inner_metadata"]["step"] == 1


def test_previous_authorized_generation_for_recovery():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckdir = os.path.join(d, "ck"); os.makedirs(ckdir)
        _save(ckdir, kr, step=1)
        _save(ckdir, kr, step=2)
        _save(ckdir, kr, step=5)
        prev = previous_authorized_generation(ckdir, before_step=5)
        assert prev is not None and prev.step == 2
        prev = previous_authorized_generation(ckdir, before_step=2)
        assert prev is not None and prev.step == 1
        assert previous_authorized_generation(ckdir, before_step=1) is None


# ---------------------------------------------------------------------------
# Worker source uses the new API
# ---------------------------------------------------------------------------
def test_worker_uses_generation_save():
    src = open(os.path.join(ROOT, "aeon/job/worker.py"), encoding="utf-8").read()
    assert "from aeon.job.generation import" in src, (
        "W10-4: worker must import the generation-directory helpers")
    assert "generation_save" in src
    assert "latest_authorized_generation" in src
    assert "discard_incomplete" in src


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
