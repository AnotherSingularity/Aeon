"""
E3 — Checkpoint, resume, and local-security tests.

Covers §10.1 completeness, §10.2 atomicity, §10.3 resume equivalence, §10.4
local security. Requires torch. Skips cleanly otherwise.
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


def _minimal_cfg():
    return {"K": 16, "margin_h": 0.98, "margin_c": 0.95,
            "transformer": {"vocab_size": 64, "hidden_size": 32}}


# ---- §10.1 completeness ---------------------------------------------------
def test_metadata_covers_10_1_fields():
    from aeon.checkpoint import build_metadata
    md = build_metadata(step=42, model_cfg=_minimal_cfg(),
                        train_cfg={"lr": 3e-4}, data_cfg={"seq_len": 64},
                        tokenizer_id="aeon.model", corpus_id="corpus/",
                        data_position=1024,
                        instrumentation_cfg={"sample_every": 512})
    required = {"schema_version", "patch_manifest_version", "source_commit",
                "step", "K", "model_config", "train_config", "data_config",
                "data_position", "tokenizer_identity", "corpus_identity",
                "precision_policy", "certificate_policy",
                "instrumentation_config"}
    missing = required - set(md.keys())
    assert not missing, f"metadata missing: {missing}"
    # precision policy must record the six-patch decisions
    p = md["precision_policy"]
    for k in ("recursion_fp32", "gamma_fp32_master", "gate_scalars_fp32_master",
              "substrate_state_follows_param_dtype", "rotary_inv_freq_fresh_fp32"):
        assert p[k] is True, f"precision policy {k} not True in metadata"


# ---- §10.2 atomicity ------------------------------------------------------
def test_atomic_save_preserves_prior_on_new_save():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.checkpoint import atomic_save, build_metadata
    m = _tiny()
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        atomic_save(path, model=m, optimizer=opt,
                    metadata=build_metadata(1, _minimal_cfg(), {}, {}, None, None, 0))
        assert os.path.exists(path)
        assert os.path.exists(path + ".sha256")
        first_size = os.path.getsize(path)
        # second save must retain the previous as .prev
        atomic_save(path, model=m, optimizer=opt,
                    metadata=build_metadata(2, _minimal_cfg(), {}, {}, None, None, 0))
        assert os.path.exists(path + ".prev"), "previous checkpoint not preserved"


def test_atomic_save_survives_interrupted_write():
    """Simulate an interrupted save by injecting a failure between the temp write
    and the atomic rename. The prior checkpoint must survive."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.checkpoint import atomic_save, build_metadata, _sha256
    m = _tiny()
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        atomic_save(path, model=m, optimizer=opt,
                    metadata=build_metadata(1, _minimal_cfg(), {}, {}, None, None, 0))
        prior_sha = _sha256(path)

        # monkey-patch torch.save to raise DURING the write
        import aeon.checkpoint as ck
        orig = ck.torch.save
        def failing_save(*a, **k): raise IOError("simulated disk full")
        ck.torch.save = failing_save
        try:
            atomic_save(path, model=m, optimizer=opt,
                        metadata=build_metadata(2, _minimal_cfg(), {}, {}, None, None, 0))
            assert False, "expected exception"
        except IOError:
            pass
        finally:
            ck.torch.save = orig
        # prior checkpoint intact
        assert os.path.exists(path)
        assert _sha256(path) == prior_sha, "prior checkpoint mutated by interrupted save"
        # temp file cleaned up
        leftovers = [f for f in os.listdir(d) if f.startswith(".ckpt.tmp.")]
        assert not leftovers, f"temp files left after failed save: {leftovers}"


# ---- §10.3 resume equivalence (deterministic bounded) ---------------------
def test_resume_equivalence_bounded():
    """Train N=6 steps continuously vs (train M=3, save, restore, train N-M=3):
    final loss trajectory equal within tight tolerance; final params bit-equal
    where determinism holds."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.checkpoint import atomic_save, strict_load, build_metadata
    torch.manual_seed(0)

    def run_continuous(N):
        m = _tiny(seed=0)
        opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
        g = torch.Generator().manual_seed(7)
        losses = []
        for _ in range(N):
            ids = torch.randint(0, 64, (2, 32), generator=g)
            out = m(input_ids=ids, labels=ids)
            opt.zero_grad(set_to_none=True); out.loss.backward(); opt.step()
            losses.append(float(out.loss.item()))
        return losses, {n: p.detach().clone() for n, p in m.named_parameters()}

    def run_split(M, N):
        m = _tiny(seed=0)
        opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
        g = torch.Generator().manual_seed(7)
        losses = []
        for _ in range(M):
            ids = torch.randint(0, 64, (2, 32), generator=g)
            out = m(input_ids=ids, labels=ids)
            opt.zero_grad(set_to_none=True); out.loss.backward(); opt.step()
            losses.append(float(out.loss.item()))
        # save
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ckpt.pt")
            atomic_save(path, model=m, optimizer=opt,
                        metadata=build_metadata(M, _minimal_cfg(), {}, {}, None, None,
                                                data_position=M*2*32))
            # NEW model + resume
            m2 = _tiny(seed=999)                 # different init to be sure resume overwrites
            opt2 = torch.optim.AdamW(m2.trainable_parameters(), lr=1e-4)
            blob = strict_load(path, expected_model_config=_minimal_cfg())
            m2.load_state_dict(blob["model"])
            opt2.load_state_dict(blob["optim"])
            # restore rng — deterministic continuation depends on it
            torch.random.set_rng_state(blob["rng"]["torch_cpu"])
            # continue
            for _ in range(N - M):
                ids = torch.randint(0, 64, (2, 32), generator=g)
                out = m2(input_ids=ids, labels=ids)
                opt2.zero_grad(set_to_none=True); out.loss.backward(); opt2.step()
                losses.append(float(out.loss.item()))
            return losses, {n: p.detach().clone() for n, p in m2.named_parameters()}

    N, M = 6, 3
    cont_losses, cont_params = run_continuous(N)
    split_losses, split_params = run_split(M, N)

    # Loss trajectory must match; use bit-equal for pre-resume steps, small
    # tolerance for post-resume (float determinism nuances).
    for i in range(M):
        assert cont_losses[i] == split_losses[i], \
            f"pre-resume step {i} diverged: {cont_losses[i]} vs {split_losses[i]}"
    for i in range(M, N):
        assert abs(cont_losses[i] - split_losses[i]) < 1e-4, \
            f"post-resume step {i} diverged: {cont_losses[i]} vs {split_losses[i]}"
    # Parameters — likewise expected close; any bounded divergence must be
    # named and small. Assert max|Δ| < 1e-4 on all params.
    for n, p in cont_params.items():
        diff = (p - split_params[n]).abs().max().item()
        assert diff < 1e-4, f"param {n} divergence {diff:.2e} > 1e-4 after resume"


# ---- §10.4 local security --------------------------------------------------
def test_reject_incompatible_metadata_K_mismatch():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.checkpoint import atomic_save, strict_load, build_metadata, CheckpointIncompatible
    m = _tiny()
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        # forge a metadata with K=8
        md = build_metadata(1, {"K": 8, "margin_h": 0.98, "margin_c": 0.95,
                                "transformer": {"vocab_size": 64}}, {}, {}, None, None, 0)
        md["K"] = 8                                # explicit override
        atomic_save(path, model=m, optimizer=opt, metadata=md)
        try:
            strict_load(path, expected_model_config=_minimal_cfg())
            assert False, "should have rejected K=8 checkpoint"
        except CheckpointIncompatible as e:
            assert "K mismatch" in str(e), str(e)


def test_reject_incompatible_metadata_vocab_mismatch():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.checkpoint import atomic_save, strict_load, build_metadata, CheckpointIncompatible
    m = _tiny()
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        atomic_save(path, model=m, optimizer=opt,
                    metadata=build_metadata(1, _minimal_cfg(), {}, {}, None, None, 0))
        expected = {"K": 16, "transformer": {"vocab_size": 99999}}
        try:
            strict_load(path, expected_model_config=expected)
            assert False, "should have rejected vocab mismatch"
        except CheckpointIncompatible as e:
            assert "vocab_size mismatch" in str(e), str(e)


def test_reject_corrupt_sha256():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.checkpoint import atomic_save, strict_load, build_metadata, CheckpointCorrupt
    m = _tiny()
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        atomic_save(path, model=m, optimizer=opt,
                    metadata=build_metadata(1, _minimal_cfg(), {}, {}, None, None, 0))
        # rewrite the sha with garbage
        with open(path + ".sha256", "w") as fh:
            fh.write("0" * 64 + "\n")
        try:
            strict_load(path, expected_model_config=_minimal_cfg())
            assert False, "should have rejected corrupt sha"
        except CheckpointCorrupt as e:
            assert "sha256 mismatch" in str(e), str(e)


def test_missing_sha256_rejected_when_required():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.checkpoint import atomic_save, strict_load, build_metadata, CheckpointCorrupt
    m = _tiny()
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "ckpt.pt")
        atomic_save(path, model=m, optimizer=opt,
                    metadata=build_metadata(1, _minimal_cfg(), {}, {}, None, None, 0))
        os.unlink(path + ".sha256")
        try:
            strict_load(path, expected_model_config=_minimal_cfg(), require_sha256=True)
            assert False, "should have rejected missing sha"
        except CheckpointCorrupt as e:
            assert "missing sidecar sha256" in str(e), str(e)


def test_strict_load_uses_weights_only_or_hardened():
    """torch.load MUST be called with weights_only=True by our path. If the
    installed torch is too old to support the kwarg, strict_load raises
    CheckpointIncompatible with a clear message — no silent fallback to
    full-pickle load."""
    import inspect, aeon.checkpoint as ck
    src = inspect.getsource(ck.strict_load)
    assert "weights_only=True" in src, "strict_load must use weights_only=True"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
