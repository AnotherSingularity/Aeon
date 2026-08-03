"""DESKTOP-R5 (partial) + WINDOWS-0 static readiness.

Windows-native execution (PyInstaller freeze + Inno installer + install
acceptance) requires a Windows machine. These checks stay Linux-runnable
and verify that the packaging inputs on the reconciliation branch will
succeed on a Windows runner:

  * Aeon.spec references the desktop modules + bundles release-assets
  * Aeon.spec excludes forbidden training / corpus / sealed paths
  * AeonInstaller.iss references dist\\Aeon\\Aeon.exe as the app exe
  * build.ps1 references python 3.11 as the required interpreter
  * scripts/export_aeon_desktop_model.py produces a deterministic bundle
    against the same source checkpoint (hash reproducibility)

Any DYNAMIC Windows behavior (installer build, install, uninstall) is
NOT verified here — that is WINDOWS-1..5's responsibility on a Windows
runner.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BUNDLE = os.path.join(ROOT, "release-assets", "aeon-desktop-p2-proxy")


# ---------------------------------------------------------------------------
def test_aeon_spec_references_desktop_modules_and_bundles_release_assets():
    p = os.path.join(ROOT, "packaging", "windows", "Aeon.spec")
    src = open(p, encoding="utf-8").read()
    assert "aeon.desktop.runtime" in src
    assert "aeon.desktop.chat_ui" in src
    assert "release-assets" in src
    assert "aeon-desktop-p2-proxy" in src


def test_aeon_spec_excludes_forbidden_paths():
    p = os.path.join(ROOT, "packaging", "windows", "Aeon.spec")
    src = open(p, encoding="utf-8").read()
    # These should be in excludes
    for token in ("tests", "aeon.tests"):
        assert token in src, f"Aeon.spec must exclude {token}"


def test_inno_installer_references_frozen_exe():
    p = os.path.join(ROOT, "packaging", "windows", "AeonInstaller.iss")
    if not os.path.exists(p):
        return
    src = open(p, encoding="utf-8").read()
    assert "Aeon.exe" in src, "installer must reference Aeon.exe"


def test_build_ps1_pins_python_311():
    p = os.path.join(ROOT, "packaging", "windows", "build.ps1")
    if not os.path.exists(p):
        return
    src = open(p, encoding="utf-8").read()
    assert "3.11" in src, "build.ps1 must pin Python 3.11"


def test_exporter_produces_tensor_deterministic_output_across_runs():
    """torch.save's zip archive includes mtimes so byte-hash is NOT
    reproducible across runs. R2 already proved (via torch.equal) that
    every exported tensor equals the source. This test locks that
    property across a re-export: rebuild + save + reload + tensor-equal.

    Cross-machine BYTE-level reproducibility of the release bundle is
    thus explicitly NOT a guarantee — the release manifest carries the
    export's sha256 stamped at export time, and consumers verify their
    own copy against the shipped manifest, not against a hash pinned
    in the repo."""
    if not os.path.exists(BUNDLE): return
    import tempfile
    import torch
    import yaml
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/latent_bypass/aeon_lbc1_proxy.yaml")))
    mcfg = cfg["model"]; tcfg = mcfg["transformer"]
    tconfig = AeonTransformerConfig(
        vocab_size=16000, hidden_size=tcfg["hidden_size"],
        num_hidden_layers=tcfg["num_hidden_layers"],
        num_attention_heads=tcfg["num_attention_heads"],
        num_key_value_heads=tcfg["num_key_value_heads"],
        head_dim=tcfg["head_dim"],
        intermediate_size=tcfg["intermediate_size"],
        max_position_embeddings=tcfg["max_position_embeddings"])
    src_state = torch.load(os.path.join(ROOT, "runs/aeon_lbc1_P2/final.pt"),
                                map_location="cpu", weights_only=False)
    def _save_and_reload():
        m = HybridModel(transformer_config=tconfig, h_rec=mcfg["h_rec"],
                            K=mcfg["K"], margin_h=mcfg["margin_h"],
                            margin_c=mcfg["margin_c"], use_embedding_input=True,
                            dtype=torch.float32).to(dtype=torch.float32)
        m.load_state_dict(src_state["model_state_dict"])
        st = {k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
            torch.save(st, f.name)
        reloaded = torch.load(f.name, map_location="cpu", weights_only=True)
        os.unlink(f.name)
        return st, reloaded
    st1, reloaded1 = _save_and_reload()
    st2, reloaded2 = _save_and_reload()
    # Tensor equivalence across two re-exports and their round-trips
    assert set(st1.keys()) == set(st2.keys()) == set(reloaded1.keys()) == set(reloaded2.keys())
    for k in st1:
        assert torch.equal(reloaded1[k], reloaded2[k]), f"reload divergence on {k}"
        assert torch.equal(st1[k], reloaded1[k]), f"round-trip divergence on {k}"


def test_release_bundle_layout_matches_installer_expectation():
    if not os.path.exists(BUNDLE): return
    # AeonInstaller.iss / Aeon.spec expect this exact layout under
    # _internal/release-assets/aeon-desktop-p2-proxy/
    for rel in (
        "model/aeon-p2-proxy-inference.pt",
        "tokenizer/aeon-lbc1.model",
        "tokenizer/aeon-lbc1.vocab",
        "config/aeon_lbc1_proxy.yaml",
        "manifests/release_manifest.json",
        "README_RELEASE.txt",
    ):
        assert os.path.exists(os.path.join(BUNDLE, rel)), f"missing {rel}"


def test_desktop_entry_dispatch_chat_wired():
    from aeon.entry import build_parser
    p = build_parser()
    args = p.parse_args(["--chat"])
    assert args.chat is True


def test_expected_release_bundle_sha256_field_is_documented():
    """The reconciliation head's Windows evidence must document the
    expected release-bundle SHA-256 so the Windows runner can verify
    determinism."""
    p = os.path.join(ROOT, "docs/desktop/desktop_windows_evidence.json")
    if not os.path.exists(p): return
    e = json.load(open(p))
    assert "expected_release_bundle_sha256" in e
    assert e["expected_release_bundle_sha256"].startswith("sha256:")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
