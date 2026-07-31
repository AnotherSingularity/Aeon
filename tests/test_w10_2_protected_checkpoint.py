"""W10-2 — the GUI worker's checkpoint path is now the F3 protected envelope.

Every test in this file drives a real (tiny) HybridModel + AdamW through
``aeon.protected_checkpoint.protected_save`` and back via ``protected_load``
under the worker's key-store convention. The tests do not spawn the worker
subprocess; instead they exercise the exact code paths ``_save_checkpoint``
and the resume block in ``_run_training_loop`` invoke, so a bug in the
save/load contract shows up here.

Covers:

* The saved artifact carries a ``.meta.json`` with a valid ``mac_hex`` tag.
* ``protected_load`` accepts the artifact under the same job HMAC key.
* Tampering the payload bytes -> ``CheckpointAuthenticationError``.
* Tampering the meta bytes -> ``CheckpointAuthenticationError``.
* A different HMAC key -> ``CheckpointAuthenticationError``.
* ``ensure_job_hmac_keyref(allow_create=False)`` on a fresh job dir refuses.
* Anti-rollback: attempted resume from an earlier authorized_step fails
  without a ``RecoveryDecision`` and succeeds with one.
* The per-job key file has 32 bytes and (on POSIX) restrictive permissions.
* No plaintext HMAC key surfaces in checkpoint contents.
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
from aeon.job.key_store import (
    ensure_job_hmac_keyref, KeyStoreError, HMAC_KEY_LEN, KEY_FILENAME,
)
from aeon.protected_checkpoint import (
    protected_save, protected_load, RecoveryDecision,
    CheckpointAuthenticationError, AntiRollbackViolation,
)


# ---------------------------------------------------------------------------
def _tiny_model():
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
    fb = getattr(m.substrate, "feedback", None)
    if fb is not None and isinstance(fb.gate_alpha, torch.nn.Parameter):
        fb.gate_alpha.data = fb.gate_alpha.data.float()
        fb.gate_threshold.data = fb.gate_threshold.data.float()
    return m, tcfg


def _save_one(job_dir, ckpt_path, step, keyref, mcfg=None):
    model, tcfg = _tiny_model()
    opt = torch.optim.AdamW(model.trainable_parameters(), lr=1e-3)
    mcfg = mcfg or {"K": 16, "transformer": tcfg.__dict__}
    md = build_metadata(step=step, model_cfg=mcfg,
                         train_cfg={"lr": 1e-3, "batch_size": 1, "seed": 0},
                         data_cfg={"seq_len": 8},
                         tokenizer_id="sha256:test-tok",
                         corpus_id="sha256:test-corpus",
                         data_position=step * 8,
                         instrumentation_cfg={"sample_every": 512,
                                              "enabled": True})
    protected_save(ckpt_path, model=model, optimizer=opt, metadata=md,
                    keyref_mac=keyref, authorized_step=step)
    return model, mcfg


# ---------------------------------------------------------------------------
# Key-store contract
# ---------------------------------------------------------------------------
def test_key_store_creates_32_bytes_on_first_call():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d, allow_create=True)
        kp = os.path.join(d, KEY_FILENAME)
        assert os.path.exists(kp), "key file must exist after first ensure"
        assert len(Path(kp).read_bytes()) == HMAC_KEY_LEN
        assert kr.key_bytes() == Path(kp).read_bytes()


def test_key_store_reuses_existing_key():
    with tempfile.TemporaryDirectory() as d:
        kr1 = ensure_job_hmac_keyref(d, allow_create=True)
        k1 = kr1.key_bytes()
        kr2 = ensure_job_hmac_keyref(d, allow_create=True)
        assert kr2.key_bytes() == k1, "second call must return the same key"


def test_key_store_refuses_create_when_disallowed():
    with tempfile.TemporaryDirectory() as d:
        try:
            ensure_job_hmac_keyref(d, allow_create=False)
            raise AssertionError("expected KeyStoreError")
        except KeyStoreError:
            pass


def test_key_file_permissions_on_posix():
    if os.name == "nt":
        return  # per-user LOCALAPPDATA suffices on Windows
    with tempfile.TemporaryDirectory() as d:
        ensure_job_hmac_keyref(d, allow_create=True)
        kp = os.path.join(d, KEY_FILENAME)
        mode = os.stat(kp).st_mode & 0o777
        assert mode == 0o600, f"key file mode should be 0600, got {oct(mode)}"


# ---------------------------------------------------------------------------
# Protected save/load round-trip
# ---------------------------------------------------------------------------
def test_protected_save_round_trip():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d, allow_create=True)
        ckpt = os.path.join(d, "ckpt_10.pt")
        _save_one(d, ckpt, step=10, keyref=kr)
        assert os.path.exists(ckpt)
        assert os.path.exists(ckpt + ".meta.json")
        assert os.path.exists(ckpt + ".sha256")
        meta = json.load(open(ckpt + ".meta.json"))
        assert meta["mac_algo"] == "hmac-sha256"
        assert isinstance(meta.get("mac_hex"), str) and len(meta["mac_hex"]) == 64
        assert meta["authorized_step"] == 10

        # Load — must succeed with the SAME key.
        blob = protected_load(ckpt, keyref_mac=kr,
                               expected_model_config={
                                   "K": 16, "transformer": {"vocab_size": 32}})
        assert blob["envelope_metadata"]["inner_metadata"]["step"] == 10
        assert blob["envelope_metadata"]["inner_metadata"]["data_position"] == 80


def test_tampered_payload_bytes_fail_authentication():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckpt = os.path.join(d, "ckpt_5.pt")
        _save_one(d, ckpt, step=5, keyref=kr)
        # Corrupt one byte of the payload
        b = bytearray(Path(ckpt).read_bytes())
        b[100] ^= 0xFF
        Path(ckpt).write_bytes(b)
        # The sha256 sidecar becomes stale first; then the MAC also fails.
        # Either failure lives under CheckpointCorrupt-family exceptions.
        try:
            protected_load(ckpt, keyref_mac=kr,
                            expected_model_config={
                                "K": 16, "transformer": {"vocab_size": 32}})
            raise AssertionError("expected failure on tampered payload")
        except Exception as e:
            # sha256 mismatch (CheckpointCorrupt) or MAC failure — both accepted
            assert "mismatch" in str(e).lower() or "mac" in str(e).lower(), str(e)


def test_tampered_metadata_json_fails_authentication():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckpt = os.path.join(d, "ckpt_3.pt")
        _save_one(d, ckpt, step=3, keyref=kr)
        # Rewrite meta.json with a modified authorized_step but keep the MAC.
        meta_path = ckpt + ".meta.json"
        meta = json.load(open(meta_path))
        meta["authorized_step"] = meta["authorized_step"] + 5
        with open(meta_path, "w") as fh:
            json.dump(meta, fh)
        try:
            protected_load(ckpt, keyref_mac=kr,
                            expected_model_config={
                                "K": 16, "transformer": {"vocab_size": 32}})
            raise AssertionError("expected MAC failure on tampered meta")
        except CheckpointAuthenticationError:
            pass


def test_wrong_key_fails_authentication():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        ckpt = os.path.join(d, "ckpt_2.pt")
        _save_one(d, ckpt, step=2, keyref=kr)
        # Use a fresh key from a different job dir
        with tempfile.TemporaryDirectory() as d2:
            kr2 = ensure_job_hmac_keyref(d2)
            try:
                protected_load(ckpt, keyref_mac=kr2,
                                expected_model_config={
                                    "K": 16, "transformer": {"vocab_size": 32}})
                raise AssertionError("expected MAC failure under wrong key")
            except CheckpointAuthenticationError:
                pass


# ---------------------------------------------------------------------------
# Anti-rollback (F3.3)
# ---------------------------------------------------------------------------
def test_anti_rollback_refuses_older_checkpoint_without_decision():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        older = os.path.join(d, "ckpt_5.pt")
        _save_one(d, older, step=5, keyref=kr)
        # Suppose the running authorized state is at step 20 — resuming from
        # step 5 should be refused.
        try:
            protected_load(older, keyref_mac=kr,
                            expected_model_config={
                                "K": 16, "transformer": {"vocab_size": 32}},
                            current_authorized_step=20)
            raise AssertionError("expected anti-rollback refusal")
        except AntiRollbackViolation:
            pass


def test_anti_rollback_allows_older_checkpoint_with_recovery_decision():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        older = os.path.join(d, "ckpt_5.pt")
        _save_one(d, older, step=5, keyref=kr)
        rd = RecoveryDecision(
            operator_authorization_ref="op:test",
            reason="corrupted current generation",
            current_state_identity="sha256:current",
            selected_state_identity="sha256:selected",
            integrity_result="verified",
            recovery_policy_version=1,
            resulting_authorized_state=5)
        blob = protected_load(older, keyref_mac=kr,
                               expected_model_config={
                                   "K": 16, "transformer": {"vocab_size": 32}},
                               current_authorized_step=20,
                               recovery_decision=rd)
        assert blob["envelope_metadata"]["accepted_via_recovery_decision"]


# ---------------------------------------------------------------------------
# Worker source: A4/A5 flipped
# ---------------------------------------------------------------------------
def test_worker_uses_protected_save_not_atomic_save():
    src = open(os.path.join(ROOT, "aeon/job/worker.py"), encoding="utf-8").read()
    assert "protected_save" in src, (
        "W10-2: worker._save_checkpoint must call protected_save")
    # atomic_save may still appear as a string inside a comment or in the
    # import list of another module, but it must not be called directly from
    # aeon/job/worker.py's code lines.
    import re
    code_lines = [line for line in src.splitlines()
                    if not line.lstrip().startswith("#")]
    body_code = "\n".join(code_lines)
    assert "atomic_save(" not in body_code, (
        "W10-2: worker must not call atomic_save directly (use protected_save)")
    assert "protected_load" in body_code, (
        "W10-2: worker resume must use protected_load")
    assert "strict_load(" not in body_code, (
        "W10-2: worker must not fall back to strict_load")


def test_worker_uses_ensure_job_hmac_keyref():
    src = open(os.path.join(ROOT, "aeon/job/worker.py"), encoding="utf-8").read()
    assert "from aeon.job.key_store import ensure_job_hmac_keyref" in src
    assert "ensure_job_hmac_keyref(job.job_dir" in src


# ---------------------------------------------------------------------------
# HMAC key never surfaces in checkpoint contents
# ---------------------------------------------------------------------------
def test_hmac_key_bytes_do_not_appear_in_checkpoint_or_meta():
    with tempfile.TemporaryDirectory() as d:
        kr = ensure_job_hmac_keyref(d)
        key_hex = kr.key_bytes().hex()
        ckpt = os.path.join(d, "ckpt_1.pt")
        _save_one(d, ckpt, step=1, keyref=kr)
        for path in (ckpt, ckpt + ".meta.json", ckpt + ".sha256"):
            data = Path(path).read_bytes()
            assert key_hex.encode("utf-8") not in data, (
                f"HMAC key hex appears in {path!r}")
            # And the raw bytes shouldn't appear either
            assert kr.key_bytes() not in data, (
                f"HMAC key raw bytes appear in {path!r}")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
