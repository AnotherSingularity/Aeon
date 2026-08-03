"""scripts/export_aeon_desktop_model.py — DESKTOP-1 inference-only exporter.

Loads the protected P2 training checkpoint, extracts only inference-required
state, and writes a self-contained release bundle under
    release-assets/aeon-desktop-p2-proxy/

The release bundle is what the desktop packages ship. The exporter never
modifies the protected checkpoint, and it never emits training material
(optimizer state, corpus, sealed test, or L3-L5 telemetry).

Serialization:
  * safetensors is preferred but is not installed in this environment.
  * The exporter therefore writes a bare state_dict via torch.save that
    is safe to reload with torch.load(..., weights_only=True): the file
    contains only tensors under string keys — no custom classes.
  * The desktop runtime MUST load this file with weights_only=True and
    reject any file that requires arbitrary deserialization.

Bundle layout (per §8):
    release-assets/aeon-desktop-p2-proxy/
        model/aeon-p2-proxy-inference.pt          (state_dict only)
        tokenizer/aeon-lbc1.model
        tokenizer/aeon-lbc1.vocab
        config/aeon_lbc1_proxy.yaml
        manifests/model_manifest.json
        manifests/tokenizer_manifest.json
        manifests/architecture_manifest.json
        manifests/release_manifest.json
        licenses/NOTICE.txt
        README_RELEASE.txt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aeon.hybrid import HybridModel
from aeon.transformer import AeonTransformerConfig


# Fixed inputs (paths + expected identities)
P2_CHECKPOINT = ROOT / "runs" / "aeon_lbc1_P2" / "final.pt"
CONFIG = ROOT / "configs" / "latent_bypass" / "aeon_lbc1_proxy.yaml"
TOK_MODEL = ROOT / "research-data" / "AEON-LBC-1" / "tokenizer" / "aeon-lbc1.model"
TOK_VOCAB = ROOT / "research-data" / "AEON-LBC-1" / "tokenizer" / "aeon-lbc1.vocab"

EXPECTED_CKPT_SHA = "sha256:962fcd5e65a88e3b6d061e73968bea7eb3581c4d8a15b49be82427004db9fc3c"
EXPECTED_TOK_SHA = "sha256:064ab6a98ee4b8177249f14dc09e60ccaa9986b66cb84a4072c68dc4de533481"
EXPECTED_K = 16
EXPECTED_VOCAB = 16000
EXPECTED_PARAMS = 7015366


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def build_model(cfg: dict, vocab_size: int, dtype=torch.float32):
    mcfg = cfg["model"]
    tcfg = mcfg["transformer"]
    tconfig = AeonTransformerConfig(
        vocab_size=vocab_size,
        hidden_size=tcfg["hidden_size"],
        num_hidden_layers=tcfg["num_hidden_layers"],
        num_attention_heads=tcfg["num_attention_heads"],
        num_key_value_heads=tcfg["num_key_value_heads"],
        head_dim=tcfg["head_dim"],
        intermediate_size=tcfg["intermediate_size"],
        max_position_embeddings=tcfg["max_position_embeddings"],
    )
    m = HybridModel(
        transformer_config=tconfig,
        h_rec=mcfg["h_rec"],
        K=mcfg["K"],
        margin_h=mcfg["margin_h"],
        margin_c=mcfg["margin_c"],
        use_embedding_input=True,
        dtype=dtype,
    )
    return m.to(dtype=dtype)


def digest_config(cfg: dict) -> str:
    """Digest of the canonical JSON of the config's model section — used
    as configuration identity."""
    b = json.dumps(cfg["model"], sort_keys=True).encode("utf-8")
    return "sha256:" + hashlib.sha256(b).hexdigest()


def digest_architecture(model: HybridModel) -> str:
    """Digest over the sorted (name, shape, dtype) triples of every parameter
    — captures architectural identity independent of weights."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode("utf-8"))
        h.update(b"|")
        h.update(str(tuple(p.shape)).encode("utf-8"))
        h.update(b"|")
        h.update(str(p.dtype).encode("utf-8"))
        h.update(b"\n")
    return "sha256:" + h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(ROOT / "release-assets" / "aeon-desktop-p2-proxy"))
    ap.add_argument("--release-id", default=None,
                     help="Release id (default derived from checkpoint+config).")
    ap.add_argument("--release-label", default="Aeon Desktop — Research Preview (7M P2 Proxy)")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()

    # 1-6. Verify inputs are the authenticated identities we expect.
    ckpt_sha = sha256_file(P2_CHECKPOINT)
    tok_sha = sha256_file(TOK_MODEL)
    if ckpt_sha != EXPECTED_CKPT_SHA:
        print(f"REFUSE: P2 checkpoint sha mismatch\n  got={ckpt_sha}\n  want={EXPECTED_CKPT_SHA}",
                file=sys.stderr); return 2
    if tok_sha != EXPECTED_TOK_SHA:
        print(f"REFUSE: tokenizer sha mismatch\n  got={tok_sha}\n  want={EXPECTED_TOK_SHA}",
                file=sys.stderr); return 3

    # 7. K = 16
    ckpt_state = torch.load(P2_CHECKPOINT, map_location="cpu", weights_only=False)
    if int(ckpt_state.get("K", -1)) != EXPECTED_K:
        print(f"REFUSE: fixed_k mismatch: {ckpt_state.get('K')}", file=sys.stderr); return 4

    # 8-11. Instantiate authoritative model + strict load
    cfg = yaml.safe_load(open(CONFIG))
    if int(cfg["model"]["K"]) != EXPECTED_K:
        print(f"REFUSE: config K != 16", file=sys.stderr); return 5

    model = build_model(cfg, vocab_size=EXPECTED_VOCAB, dtype=torch.float32)
    n_params = sum(p.numel() for p in model.parameters())
    if n_params != EXPECTED_PARAMS:
        print(f"REFUSE: parameter count {n_params} != {EXPECTED_PARAMS}", file=sys.stderr); return 6

    missing, unexpected = model.load_state_dict(ckpt_state["model_state_dict"], strict=False)
    if missing:
        print(f"REFUSE: missing keys in checkpoint: {missing[:5]}...", file=sys.stderr); return 7
    if unexpected:
        print(f"REFUSE: unexpected keys in checkpoint: {unexpected[:5]}...", file=sys.stderr); return 8

    # 13-22. Extract inference-only state. Only the model.state_dict()
    # — no optimizer, scheduler, scaler, corpus cursor, or training
    # counters. torch.save on a dict of {str: tensor} produces a file
    # that is safe under weights_only=True (no custom classes).
    inference_state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}

    # Layout
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "model").mkdir(exist_ok=True)
    (out_root / "tokenizer").mkdir(exist_ok=True)
    (out_root / "config").mkdir(exist_ok=True)
    (out_root / "manifests").mkdir(exist_ok=True)
    (out_root / "licenses").mkdir(exist_ok=True)

    model_out_path = out_root / "model" / "aeon-p2-proxy-inference.pt"

    # 23. Write to a temporary file. 24. Reread and validate. 25. Atomically promote.
    with tempfile.NamedTemporaryFile(
            dir=str(model_out_path.parent), delete=False,
            suffix=".tmp") as tmp:
        tmp_path = Path(tmp.name)
    try:
        torch.save(inference_state, tmp_path)
        # Re-read strictly (weights_only=True): rejects arbitrary classes.
        reread = torch.load(str(tmp_path), map_location="cpu", weights_only=True)
        if set(reread.keys()) != set(inference_state.keys()):
            print("REFUSE: re-read key set differs", file=sys.stderr); return 9
        for k in inference_state:
            if reread[k].shape != inference_state[k].shape:
                print(f"REFUSE: re-read shape differs for {k}", file=sys.stderr); return 10
            if not torch.equal(reread[k], inference_state[k]):
                print(f"REFUSE: re-read tensor bytes differ for {k}", file=sys.stderr); return 11
        os.replace(str(tmp_path), str(model_out_path))
    except Exception:
        try: os.unlink(str(tmp_path))
        except FileNotFoundError: pass
        raise

    # 26. Compute SHA-256 of the final export.
    model_export_sha = sha256_file(model_out_path)
    model_export_bytes = model_out_path.stat().st_size

    # Copy tokenizer + config + license notice.
    shutil.copyfile(TOK_MODEL, out_root / "tokenizer" / "aeon-lbc1.model")
    shutil.copyfile(TOK_VOCAB, out_root / "tokenizer" / "aeon-lbc1.vocab")
    shutil.copyfile(CONFIG, out_root / "config" / "aeon_lbc1_proxy.yaml")

    # Licenses / notices (minimum viable). Aeon proprietary code is NOT
    # placed here; only the notices needed for public-domain corpus + spm.
    (out_root / "licenses" / "NOTICE.txt").write_text(
        "Aeon Desktop — Research Preview (7M P2 Proxy).\n\n"
        "Tokenizer training corpus: AEON-LBC-1 — six Project Gutenberg\n"
        "public-domain works (U.S.). Corpus is NOT included in this bundle.\n\n"
        "Runtime dependencies (PyTorch CPU, sentencepiece) retain their\n"
        "upstream licenses in the installer's bundled licenses directory.\n",
        encoding="utf-8",
    )

    arch_digest = digest_architecture(model)
    cfg_digest = digest_config(cfg)

    now_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    release_id = args.release_id or f"aeon-desktop-p2-proxy-{model_export_sha[7:19]}"

    # Manifests (per §9)
    model_manifest = {
        "schema_version": 1,
        "model_artifact_relpath": "model/aeon-p2-proxy-inference.pt",
        "model_artifact_sha256": model_export_sha,
        "model_bytes": model_export_bytes,
        "model_format": "torch_state_dict_weights_only_safe",
        "parameter_count": n_params,
        "tested_scale": "7M proxy",
        "model_configuration_digest": cfg_digest,
        "architecture_manifest_digest": arch_digest,
        "checkpoint_source_relpath_in_repo": str(P2_CHECKPOINT.relative_to(ROOT)),
        "checkpoint_source_sha256": ckpt_sha,
        "checkpoint_stage": ckpt_state.get("stage"),
        "checkpoint_seed": ckpt_state.get("seed"),
        "checkpoint_useful_tokens": ckpt_state.get("useful_tokens"),
        "fixed_k": EXPECTED_K,
        "recursion_dtype": "float32",
        "substrate_dtype_policy": "follow_model_dtype",
    }
    tok_manifest = {
        "schema_version": 1,
        "tokenizer_model_relpath": "tokenizer/aeon-lbc1.model",
        "tokenizer_model_sha256": tok_sha,
        "tokenizer_vocab_relpath": "tokenizer/aeon-lbc1.vocab",
        "tokenizer_vocab_sha256": sha256_file(TOK_VOCAB),
        "tokenizer_vocab_size": EXPECTED_VOCAB,
        "special_token_ids": {"pad_id": 0, "unk_id": 1, "bos_id": 2, "eos_id": 3},
        "tokenizer_training_partition_identity": (
            "AEON-LBC-1 train (sha256:5c33cbcbe0e4ca6ad84bd6d27a751f1791e504945f82eee108f27ba4d7b07c59)"),
        "implementation": "sentencepiece_unigram_byte_fallback",
    }
    arch_manifest = {
        "schema_version": 1,
        "architecture_digest": arch_digest,
        "model_class": "aeon.hybrid.HybridModel",
        "K": EXPECTED_K,
        "config_relpath": "config/aeon_lbc1_proxy.yaml",
        "config_digest": cfg_digest,
        "parameter_count": n_params,
        "K_immutable_invariant": True,
        "one_broadcast_per_boundary_invariant": True,
        "recursion_fp32_invariant": True,
        "substrate_autonomous_gate_invariant": True,
        "six_v0_02_02_patches_intact": True,
        "ACIS_default": "OFF",
    }
    release_manifest = {
        "release_schema_version": 1,
        "release_id": release_id,
        "release_channel": "research_preview",
        "release_label": args.release_label,
        "created_from_commit": os.popen("git rev-parse HEAD").read().strip(),
        "created_at_utc": now_utc,
        "model_artifact_path": "model/aeon-p2-proxy-inference.pt",
        "model_artifact_sha256": model_export_sha,
        "model_format": "torch_state_dict_weights_only_safe",
        "parameter_count": n_params,
        "tested_scale": "7M proxy",
        "model_configuration_digest": cfg_digest,
        "architecture_manifest_digest": arch_digest,
        "checkpoint_source_identity": "runs/aeon_lbc1_P2/final.pt",
        "checkpoint_source_sha256": ckpt_sha,
        "tokenizer_model_path": "tokenizer/aeon-lbc1.model",
        "tokenizer_model_sha256": tok_sha,
        "tokenizer_vocab_path": "tokenizer/aeon-lbc1.vocab",
        "tokenizer_vocab_sha256": sha256_file(TOK_VOCAB),
        "tokenizer_vocab_size": EXPECTED_VOCAB,
        "special_token_ids": {"pad_id": 0, "unk_id": 1, "bos_id": 2, "eos_id": 3},
        "tokenizer_training_partition_identity": (
            "AEON-LBC-1 train partition"),
        "fixed_k": EXPECTED_K,
        "recursion_dtype": "float32",
        "substrate_dtype_policy": "follow_model_dtype",
        "ACIS_default": "OFF",
        "minimum_runtime_version": "aeon-desktop-runtime-1.0.0",
        "maximum_supported_runtime_version": "aeon-desktop-runtime-1.x.x",
        "network_policy": "offline_only",
        "training_code_included": False,
        "corpus_included": False,
        "sealed_test_included": False,
        "optimizer_state_included": False,
    }

    (out_root / "manifests" / "model_manifest.json").write_text(
        json.dumps(model_manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "manifests" / "tokenizer_manifest.json").write_text(
        json.dumps(tok_manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "manifests" / "architecture_manifest.json").write_text(
        json.dumps(arch_manifest, indent=2, sort_keys=True), encoding="utf-8")
    (out_root / "manifests" / "release_manifest.json").write_text(
        json.dumps(release_manifest, indent=2, sort_keys=True), encoding="utf-8")

    (out_root / "README_RELEASE.txt").write_text(
        "Aeon Desktop — Research Preview\n"
        "Model: AEON-LBC-1 P2 Proxy\n"
        f"Release id: {release_id}\n"
        "Scale: 7M parameters\n"
        "Runtime: Offline\n\n"
        "This is a bounded research preview built from the 7M-parameter P2\n"
        "proxy checkpoint. It is NOT a 350M / 1.79B / production model.\n"
        "It is NOT a Level-3-proven latent-bypass system; the research\n"
        "campaign closed at Level 2 OBSERVATIONAL_EVIDENCE with Level 3\n"
        "status = CANDIDATE_NOT_CLOSED.\n",
        encoding="utf-8",
    )

    # 27. Export report
    print(json.dumps({
        "ok": True,
        "release_id": release_id,
        "bundle_root": str(out_root),
        "model_export_sha256": model_export_sha,
        "model_bytes": model_export_bytes,
        "tokenizer_sha256": tok_sha,
        "arch_digest": arch_digest,
        "cfg_digest": cfg_digest,
        "parameter_count": n_params,
        "K": EXPECTED_K,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
