"""
F3 — Protected checkpoint envelope + anti-rollback + confidentiality tests.

Covers §F3.1 (authenticated envelope), §F3.2 (optional confidentiality),
§F3.3 (anti-rollback) and §F3.5 (compatibility / fail-closed).

Requires torch. Skips cleanly otherwise.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _have_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _tiny(seed=0):
    import torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    torch.manual_seed(seed)
    tcfg = AeonTransformerConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=64)
    m = HybridModel(h_rec=24, K=16, transformer_config=tcfg,
                    substrate={"kind": "matrix", "d_in": 24, "d_state": 24,
                               "n_head": 2, "head_size": 12},
                    dtype=torch.float32)
    m.recursion.float()
    return m


def _inner_metadata():
    return {"step": 5, "K": 16, "model_config": {"K": 16, "transformer": {"vocab_size": 64}},
            "precision_policy": {"recursion_fp32": True}, "schema_version": 1,
            "patch_manifest_version": 1}


# ---- §F3.1 authenticated envelope -------------------------------------------
def test_protected_save_load_round_trip():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.protected_checkpoint import (protected_save, protected_load,
                                            ephemeral_dev_keyref)
    m = _tiny()
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    mac_key = ephemeral_dev_keyref("mac")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck.pt")
        protected_save(path, model=m, optimizer=opt, metadata=_inner_metadata(),
                       keyref_mac=mac_key)
        assert os.path.exists(path) and os.path.exists(path + ".meta.json")
        assert os.path.exists(path + ".sha256")
        blob = protected_load(path, keyref_mac=mac_key,
                              expected_model_config={"K": 16, "transformer": {"vocab_size": 64}})
        assert "model" in blob and "envelope_metadata" in blob


def test_mac_verification_refuses_meta_tampering():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch, json
    from aeon.protected_checkpoint import (protected_save, protected_load,
                                            ephemeral_dev_keyref, CheckpointAuthenticationError)
    m = _tiny(); opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    mac_key = ephemeral_dev_keyref("mac")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck.pt")
        protected_save(path, model=m, optimizer=opt, metadata=_inner_metadata(),
                       keyref_mac=mac_key)
        meta = json.load(open(path + ".meta.json"))
        # Flip an inner-metadata field that is authenticated by the MAC
        meta["inner_metadata"]["step"] = 999
        with open(path + ".meta.json", "w") as fh:
            json.dump(meta, fh)
        try:
            protected_load(path, keyref_mac=mac_key,
                           expected_model_config={"K": 16, "transformer": {"vocab_size": 64}})
            assert False, "MAC did not detect meta tamper"
        except CheckpointAuthenticationError:
            pass


def test_wrong_mac_key_refused():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.protected_checkpoint import (protected_save, protected_load,
                                            ephemeral_dev_keyref, CheckpointAuthenticationError)
    m = _tiny(); opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    mac_key = ephemeral_dev_keyref("A")
    wrong_key = ephemeral_dev_keyref("B")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck.pt")
        protected_save(path, model=m, optimizer=opt, metadata=_inner_metadata(),
                       keyref_mac=mac_key)
        try:
            protected_load(path, keyref_mac=wrong_key,
                           expected_model_config={"K": 16, "transformer": {"vocab_size": 64}})
            assert False, "wrong MAC key accepted"
        except CheckpointAuthenticationError:
            pass


def test_one_byte_payload_tamper_refused():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.protected_checkpoint import (protected_save, protected_load,
                                            ephemeral_dev_keyref, CheckpointCorrupt,
                                            CheckpointAuthenticationError)
    m = _tiny(); opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    mac_key = ephemeral_dev_keyref("mac")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck.pt")
        protected_save(path, model=m, optimizer=opt, metadata=_inner_metadata(),
                       keyref_mac=mac_key)
        # Flip one byte in the payload
        with open(path, "r+b") as fh:
            data = bytearray(fh.read())
            data[100] ^= 0x01
            fh.seek(0); fh.write(bytes(data))
        try:
            protected_load(path, keyref_mac=mac_key,
                           expected_model_config={"K": 16, "transformer": {"vocab_size": 64}})
            assert False, "byte tamper accepted"
        except (CheckpointCorrupt, CheckpointAuthenticationError):
            pass


# ---- §F3.3 anti-rollback ---------------------------------------------------
def test_anti_rollback_refuses_older_checkpoint():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.protected_checkpoint import (protected_save, protected_load,
                                            ephemeral_dev_keyref, AntiRollbackViolation)
    m = _tiny(); opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    mac_key = ephemeral_dev_keyref("mac")
    with tempfile.TemporaryDirectory() as d:
        path_old = os.path.join(d, "old.pt")
        protected_save(path_old, model=m, optimizer=opt, metadata=_inner_metadata(),
                       keyref_mac=mac_key, authorized_step=100)
        try:
            protected_load(path_old, keyref_mac=mac_key,
                           expected_model_config={"K": 16, "transformer": {"vocab_size": 64}},
                           current_authorized_step=500)
            assert False, "anti-rollback did not fire"
        except AntiRollbackViolation:
            pass


def test_authorized_rollback_accepted_with_recovery_decision():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.protected_checkpoint import (protected_save, protected_load,
                                            ephemeral_dev_keyref, RecoveryDecision)
    m = _tiny(); opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    mac_key = ephemeral_dev_keyref("mac")
    with tempfile.TemporaryDirectory() as d:
        path_old = os.path.join(d, "old.pt")
        protected_save(path_old, model=m, optimizer=opt, metadata=_inner_metadata(),
                       keyref_mac=mac_key, authorized_step=100)
        decision = RecoveryDecision(
            operator_authorization_ref="OP-2025-001",
            reason="unit test — authorised rollback",
            current_state_identity="hash_current",
            selected_state_identity="hash_old",
            integrity_result="verified",
            recovery_policy_version=1,
            resulting_authorized_state=100,
        )
        blob = protected_load(path_old, keyref_mac=mac_key,
                              expected_model_config={"K": 16, "transformer": {"vocab_size": 64}},
                              current_authorized_step=500,
                              recovery_decision=decision)
        assert blob["envelope_metadata"]["accepted_via_recovery_decision"]["kind"] == "authorized_rollback"


# ---- §F3.2 confidentiality (optional) --------------------------------------
def test_encrypted_round_trip_or_gracefully_absent():
    """If `cryptography` is installed, round-trip an encrypted checkpoint.
    If it's not, the API must refuse (never silently fall back to plaintext)."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.protected_checkpoint import (protected_save, protected_load,
                                            ephemeral_dev_keyref, _try_aesgcm,
                                            KeyUnavailableError)
    have_aead = _try_aesgcm() is not None
    m = _tiny(); opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    mac_key = ephemeral_dev_keyref("mac"); enc_key = ephemeral_dev_keyref("enc")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck.pt")
        if have_aead:
            protected_save(path, model=m, optimizer=opt, metadata=_inner_metadata(),
                           keyref_mac=mac_key, keyref_encrypt=enc_key)
            blob = protected_load(path, keyref_mac=mac_key, keyref_encrypt=enc_key,
                                  expected_model_config={"K": 16, "transformer": {"vocab_size": 64}})
            assert "model" in blob
            # Load without the enc key must fail closed — never silent plaintext
            try:
                protected_load(path, keyref_mac=mac_key,
                               expected_model_config={"K": 16, "transformer": {"vocab_size": 64}})
                assert False, "encrypted file loaded without key"
            except KeyUnavailableError:
                pass
        else:
            try:
                protected_save(path, model=m, optimizer=opt, metadata=_inner_metadata(),
                               keyref_mac=mac_key, keyref_encrypt=enc_key)
                assert False, "encryption succeeded without AEAD backend"
            except KeyUnavailableError as e:
                assert "cryptography" in str(e)


def test_key_material_is_not_stored_in_checkpoint_or_meta():
    """Fingerprint the MAC key; assert it does not appear as bytes in either the
    payload file or the metadata JSON."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.protected_checkpoint import (protected_save, ephemeral_dev_keyref)
    m = _tiny(); opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    mac_key = ephemeral_dev_keyref("mac")
    material = mac_key.key_bytes()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ck.pt")
        protected_save(path, model=m, optimizer=opt, metadata=_inner_metadata(),
                       keyref_mac=mac_key)
        with open(path, "rb") as fh:
            assert material not in fh.read(), "MAC key bytes leaked into payload"
        with open(path + ".meta.json", "rb") as fh:
            assert material not in fh.read(), "MAC key bytes leaked into meta.json"


# ---- audit hash chain ------------------------------------------------------
def test_audit_hash_chain_verifies_and_detects_tampering():
    from aeon.audit import AuditWriter, verify_chain
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "audit.jsonl")
        w = AuditWriter(path)
        w.write("save", ckpt="ck1")
        w.write("save", ckpt="ck2")
        w.write("resume", ckpt="ck2")
        assert verify_chain(path) is None
        # Tamper: rewrite the second line's payload
        lines = open(path).readlines()
        import json
        rec = json.loads(lines[1])
        rec["payload"]["ckpt"] = "ck_wrong"
        lines[1] = json.dumps(rec, separators=(",", ":")) + "\n"
        with open(path, "w") as fh:
            fh.writelines(lines)
        err = verify_chain(path)
        assert err is not None and ("content hash" in err or "prev_hash" in err), err


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
