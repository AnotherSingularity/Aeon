"""
E4 — Offline diagnostics tests.

§11 requirements:
  * Diagnostics must not mutate the source checkpoint (byte-equal before/after).
  * Interventions are evaluation-only (in-memory hooks that are removed).
  * The tool produces a machine-readable report per checkpoint.

Requires torch. Skips cleanly otherwise.
"""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _have_torch():
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _make_ckpt(dir_, seed=0):
    """Build a tiny model + optimizer, run one step, save via atomic_save."""
    import torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    from aeon.checkpoint import atomic_save, build_metadata
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
    opt = torch.optim.AdamW(m.trainable_parameters(), lr=1e-4)
    ids = torch.randint(0, 64, (2, 32))
    out = m(input_ids=ids, labels=ids)
    opt.zero_grad(set_to_none=True); out.loss.backward(); opt.step()
    mcfg = {"K": 16, "h_rec": 24, "margin_h": 0.98, "margin_c": 0.95,
            "freeze_backbone": False, "use_embedding_input": True,
            "dtype": "float32",
            "transformer": {"vocab_size": 64, "hidden_size": 32, "intermediate_size": 64,
                            "num_hidden_layers": 2, "num_attention_heads": 2,
                            "num_key_value_heads": 1, "head_dim": 16,
                            "max_position_embeddings": 64},
            "substrate": {"kind": "matrix", "d_in": 24, "d_state": 24,
                          "n_head": 2, "head_size": 12}}
    path = os.path.join(dir_, "ckpt.pt")
    atomic_save(path, model=m, optimizer=opt,
                metadata=build_metadata(1, mcfg, {}, {"seq_len": 32}, None, None, 0))
    cfg = {"model": mcfg, "data": {"seq_len": 32}, "train": {"seed": 0}}
    cfg_path = os.path.join(dir_, "cfg.yaml")
    import yaml
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
    return cfg_path, path


def test_diagnose_all_does_not_mutate_checkpoint():
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    with tempfile.TemporaryDirectory() as d:
        cfg, ckpt = _make_ckpt(d)
        before_ckpt = _sha256(ckpt)
        before_sha = _sha256(ckpt + ".sha256")
        # invoke as a subprocess to exercise the CLI
        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run(
            [sys.executable, "scripts/diagnose.py", "--config", cfg, "--ckpt", ckpt,
             "--subcommand", "all", "--seq-len", "32"],
            capture_output=True, text=True, env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=120,
        )
        assert r.returncode == 0, f"diagnose.py failed: {r.stderr[-800:]}"
        assert _sha256(ckpt) == before_ckpt, "diagnose.py mutated the checkpoint"
        assert _sha256(ckpt + ".sha256") == before_sha, "diagnose.py mutated the sha256"

        report_path = ckpt + ".diagnostics.json"
        assert os.path.exists(report_path)
        report = json.load(open(report_path))
        assert "certificate" in report
        assert report["certificate_holds_on_load"] is True
        assert "gradients" in report
        assert "interventions" in report
        assert "feedback" in report
        # feedback report should have the 5 named results
        assert len(report["feedback"].get("results", [])) == 5


def test_diagnose_interventions_are_evaluation_only():
    """Run the interventions subcommand, then reload the checkpoint into a
    fresh model, and prove the model parameters are IDENTICAL to a control
    that never ran diagnostics (interventions did not leak into weights)."""
    if not _have_torch():
        print("  [skip] torch unavailable"); return
    import torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    from aeon.checkpoint import strict_load

    with tempfile.TemporaryDirectory() as d:
        cfg_path, ckpt = _make_ckpt(d)

        def load_params():
            torch.manual_seed(999)
            tcfg = AeonTransformerConfig(
                vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=2, num_key_value_heads=1, head_dim=16,
                max_position_embeddings=64)
            m = HybridModel(h_rec=24, K=16, transformer_config=tcfg,
                            substrate={"kind": "matrix", "d_in": 24, "d_state": 24,
                                       "n_head": 2, "head_size": 12},
                            dtype=torch.float32)
            m.recursion.float()
            mcfg = {"transformer": {"vocab_size": 64}}
            blob = strict_load(ckpt, expected_model_config=mcfg)
            m.load_state_dict(blob["model"])
            return {n: p.detach().clone() for n, p in m.named_parameters()}

        before = load_params()

        env = os.environ.copy()
        env["PYTHONPATH"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        r = subprocess.run(
            [sys.executable, "scripts/diagnose.py", "--config", cfg_path, "--ckpt", ckpt,
             "--subcommand", "interventions", "--seq-len", "32"],
            capture_output=True, text=True, env=env,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            timeout=60,
        )
        assert r.returncode == 0, r.stderr[-500:]

        after = load_params()
        for name in before:
            assert torch.equal(before[name], after[name]), \
                f"param {name} changed after diagnose interventions"


def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
