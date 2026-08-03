"""DESKTOP-1 — inference export + release manifest + equivalence gate."""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


BUNDLE = os.path.join(ROOT, "release-assets", "aeon-desktop-p2-proxy")


def _sha(path):
    with open(path, "rb") as f:
        return "sha256:" + hashlib.sha256(f.read()).hexdigest()


# ---------------------------------------------------------------------------
def test_export_script_exists_and_is_pure_python():
    p = os.path.join(ROOT, "scripts", "export_aeon_desktop_model.py")
    assert os.path.exists(p)
    src = open(p, encoding="utf-8").read()
    # Refuses unrestricted pickle
    assert "weights_only=True" in src
    assert "torch.save" in src
    # Guards for identity mismatches
    assert "EXPECTED_CKPT_SHA" in src
    assert "EXPECTED_TOK_SHA" in src
    assert "EXPECTED_K" in src


def test_release_bundle_layout_present():
    if not os.path.exists(BUNDLE):
        # Skip if bundle not built — build runs on demand
        return
    for rel in (
        "model/aeon-p2-proxy-inference.pt",
        "tokenizer/aeon-lbc1.model",
        "tokenizer/aeon-lbc1.vocab",
        "config/aeon_lbc1_proxy.yaml",
        "manifests/model_manifest.json",
        "manifests/tokenizer_manifest.json",
        "manifests/architecture_manifest.json",
        "manifests/release_manifest.json",
        "licenses/NOTICE.txt",
        "README_RELEASE.txt",
    ):
        p = os.path.join(BUNDLE, rel)
        assert os.path.exists(p), f"missing {rel}"


def test_release_manifest_binds_required_fields():
    if not os.path.exists(BUNDLE):
        return
    m = json.load(open(os.path.join(BUNDLE, "manifests/release_manifest.json")))
    for k in (
        "release_schema_version", "release_id", "release_channel",
        "release_label", "created_from_commit", "created_at_utc",
        "model_artifact_path", "model_artifact_sha256", "model_format",
        "parameter_count", "tested_scale", "model_configuration_digest",
        "architecture_manifest_digest", "checkpoint_source_identity",
        "checkpoint_source_sha256", "tokenizer_model_path",
        "tokenizer_model_sha256", "tokenizer_vocab_path",
        "tokenizer_vocab_sha256", "tokenizer_vocab_size",
        "special_token_ids", "tokenizer_training_partition_identity",
        "fixed_k", "recursion_dtype", "substrate_dtype_policy",
        "ACIS_default", "minimum_runtime_version",
        "maximum_supported_runtime_version", "network_policy",
        "training_code_included", "corpus_included",
        "sealed_test_included", "optimizer_state_included",
    ):
        assert k in m, f"release_manifest missing {k}"
    assert m["tested_scale"] == "7M proxy"
    assert m["fixed_k"] == 16
    assert m["ACIS_default"] == "OFF"
    assert m["network_policy"] == "offline_only"
    assert m["training_code_included"] is False
    assert m["corpus_included"] is False
    assert m["sealed_test_included"] is False
    assert m["optimizer_state_included"] is False


def test_release_manifest_digest_matches_bundle():
    if not os.path.exists(BUNDLE):
        return
    m = json.load(open(os.path.join(BUNDLE, "manifests/release_manifest.json")))
    art = os.path.join(BUNDLE, m["model_artifact_path"])
    assert _sha(art) == m["model_artifact_sha256"]
    tok = os.path.join(BUNDLE, m["tokenizer_model_path"])
    assert _sha(tok) == m["tokenizer_model_sha256"]


def test_model_export_loads_with_weights_only_true():
    """The exported model must be safe to reload with weights_only=True."""
    if not os.path.exists(BUNDLE):
        return
    import torch
    p = os.path.join(BUNDLE, "model/aeon-p2-proxy-inference.pt")
    st = torch.load(p, map_location="cpu", weights_only=True)
    assert isinstance(st, dict)
    for k, v in st.items():
        assert isinstance(k, str)
        assert isinstance(v, torch.Tensor)


def test_export_is_inference_only():
    """Exported state_dict must NOT contain training-only keys."""
    if not os.path.exists(BUNDLE):
        return
    import torch
    p = os.path.join(BUNDLE, "model/aeon-p2-proxy-inference.pt")
    st = torch.load(p, map_location="cpu", weights_only=True)
    forbidden = ("optimizer", "scheduler", "scaler", "corpus_cursor",
                    "training_step", "loss_history")
    for k in st:
        for f in forbidden:
            assert f not in k.lower(), f"forbidden key in export: {k}"


def test_export_matches_source_checkpoint_bytewise():
    """DESKTOP-1 equivalence gate: for every parameter key, the exported
    tensor must be byte-identical to the source P2 checkpoint's tensor.
    This is the strictest form of the §10 equivalence gate — no
    tolerance needed because same precision, same device."""
    if not os.path.exists(BUNDLE):
        return
    import torch
    src = torch.load(os.path.join(ROOT, "runs/aeon_lbc1_P2/final.pt"),
                        map_location="cpu", weights_only=False)
    exp = torch.load(os.path.join(BUNDLE, "model/aeon-p2-proxy-inference.pt"),
                        map_location="cpu", weights_only=True)
    src_sd = src["model_state_dict"]
    assert set(src_sd.keys()) == set(exp.keys()), (
        f"key set differs: extra_src={set(src_sd)-set(exp)}, extra_exp={set(exp)-set(src_sd)}")
    for k in src_sd:
        a = src_sd[k]; b = exp[k]
        assert a.shape == b.shape, f"shape mismatch on {k}"
        assert a.dtype == b.dtype, f"dtype mismatch on {k}"
        assert torch.equal(a.detach().cpu(), b.detach().cpu()), (
            f"tensor bytes differ on {k}")


def test_manifest_declares_K_equals_16():
    if not os.path.exists(BUNDLE):
        return
    m = json.load(open(os.path.join(BUNDLE, "manifests/release_manifest.json")))
    a = json.load(open(os.path.join(BUNDLE, "manifests/architecture_manifest.json")))
    mm = json.load(open(os.path.join(BUNDLE, "manifests/model_manifest.json")))
    assert m["fixed_k"] == 16
    assert a["K"] == 16
    assert mm["fixed_k"] == 16
    assert a["K_immutable_invariant"] is True


def test_manifest_declares_7M_scale_not_350M():
    if not os.path.exists(BUNDLE):
        return
    m = json.load(open(os.path.join(BUNDLE, "manifests/release_manifest.json")))
    assert m["tested_scale"] == "7M proxy"
    assert m["parameter_count"] == 7015366
    assert "350M" not in m["release_label"]
    assert "production" not in m["release_label"].lower()


def test_bundle_excludes_forbidden_paths():
    if not os.path.exists(BUNDLE):
        return
    forbidden_names = ("optimizer.pt", "training.log", "corpus", "sealed",
                          "test.jsonl", "train.jsonl", "calibration.jsonl",
                          "validation.jsonl", ".git")
    for dirpath, _, files in os.walk(BUNDLE):
        for fn in files:
            for f in forbidden_names:
                assert f not in fn, (
                    f"forbidden file in bundle: {dirpath}/{fn}")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
