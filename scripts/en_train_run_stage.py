"""scripts/en_train_run_stage.py — resumable Stage-1 / Stage-2 runner.

Usage:
    python scripts/en_train_run_stage.py --stage stage1|stage2 \\
        --corpus-package research-data/<CORPUS_ID> \\
        --seed 20260803 \\
        --lr-pilot | --lr <PEAK_LR> \\
        --out runs/en_train/<tag>/

Reads processed partitions from `<corpus_package>/processed/` and
tokenizer identity from the release-manifest section documented in
docs/en_train/EN_TRAIN_INFRASTRUCTURE.md. Refuses to start if any
architecture-invariance check fails at boot.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from aeon.hybrid import HybridModel
from aeon.transformer import AeonTransformerConfig
from aeon.tokenizer import AeonTokenizer

from aeon.en_train import (
    PROTECTED_P2_SHA256, PROTECTED_TOKENIZER_SHA256, PROTECTED_A0_DIGEST,
    STAGE1_MIX, STAGE2_MIX,
)
from aeon.en_train.proof import (
    assert_architecture_invariant, sigma_certificate,
)
from aeon.en_train.trainer import (
    DocumentBatchProvider, train_gated, run_lr_pilot, select_pilot_lr,
)


def _sha(p):
    with open(p, "rb") as f: return "sha256:" + hashlib.sha256(f.read()).hexdigest()


def build_model(cfg_path: Path, vocab_size: int):
    cfg = yaml.safe_load(open(cfg_path))
    mc = cfg["model"]; tc = mc["transformer"]
    tconfig = AeonTransformerConfig(
        vocab_size=vocab_size, hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        intermediate_size=tc["intermediate_size"],
        max_position_embeddings=tc["max_position_embeddings"])
    m = HybridModel(transformer_config=tconfig, h_rec=mc["h_rec"],
                        K=mc["K"], margin_h=mc["margin_h"],
                        margin_c=mc["margin_c"], use_embedding_input=True,
                        dtype=torch.float32).to(dtype=torch.float32)
    return m, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["stage1", "stage2"], required=True)
    ap.add_argument("--corpus-package", required=True,
                     help="Directory with processed/{train,validation,test}.jsonl")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--seed", type=int, default=20260803)
    ap.add_argument("--lr", type=float, default=None,
                     help="peak learning rate; if omitted, --lr-pilot must be set")
    ap.add_argument("--lr-pilot", action="store_true",
                     help="run the §10 pilot grid and pick the best stable LR")
    ap.add_argument("--out", required=True)
    ap.add_argument("--from-checkpoint", default="runs/aeon_lbc1_P2/final.pt")
    ap.add_argument("--config", default="configs/latent_bypass/aeon_lbc1_proxy.yaml")
    ap.add_argument("--l-native", type=int, default=256,
                     help="native training sequence capacity used for sequence buckets")
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    # ---- Freeze checks up front (§2) ----
    tok = AeonTokenizer(args.tokenizer)
    tok_sha = _sha(args.tokenizer)
    if tok_sha != PROTECTED_TOKENIZER_SHA256:
        print(f"REFUSE: tokenizer sha mismatch: got={tok_sha}",
                file=sys.stderr)
        return 2
    m, cfg = build_model(Path(args.config), vocab_size=tok.vocab_size)
    st = torch.load(args.from_checkpoint, map_location="cpu", weights_only=False)
    m.load_state_dict(st["model_state_dict"])
    # Architecture invariance
    try:
        assert_architecture_invariant(m)
    except Exception as e:
        print(f"REFUSE: architecture invariance: {e}", file=sys.stderr)
        return 3

    baseline_sigma = sigma_certificate(m)

    # ---- Data providers (assumes processed jsonl already exists) ----
    import json as _json
    def load_jsonl(path):
        streams = []
        for line in open(path, encoding="utf-8"):
            r = _json.loads(line)
            ids = tok.encode(r["text"], add_bos=False, add_eos=False)
            streams.append(ids)
        return streams

    corpus = Path(args.corpus_package)
    train_streams = load_jsonl(corpus / "processed" / "train.jsonl")
    val_streams = load_jsonl(corpus / "processed" / "validation.jsonl")
    if not train_streams or not val_streams:
        print("REFUSE: empty train or validation partition", file=sys.stderr)
        return 4
    train_provider = DocumentBatchProvider(name="train", streams=train_streams,
                                                    seed=args.seed)
    val_provider = DocumentBatchProvider(name="validation", streams=val_streams,
                                                  seed=args.seed + 1)

    # ---- LR pilot (optional) ----
    if args.lr is None and not args.lr_pilot:
        print("REFUSE: supply --lr or --lr-pilot", file=sys.stderr); return 5

    if args.lr_pilot:
        pilots = run_lr_pilot(
            model_factory=lambda: build_model(Path(args.config), tok.vocab_size)[0].to("cpu"),
            build_optimizer=lambda mm, lr: torch.optim.AdamW(mm.parameters(),
                                                                              lr=lr, weight_decay=0.01),
            train_provider=train_provider, val_provider=val_provider,
            baseline_sigma=baseline_sigma, seed=args.seed,
            batch_size=args.batch_size, seq_len=args.l_native)
        chosen = select_pilot_lr(pilots)
        if chosen is None:
            print("REFUSE: LR pilot: no LR passed stability", file=sys.stderr); return 6
        peak_lr = chosen
        print("[pilot] chose peak_lr=", peak_lr)
    else:
        peak_lr = float(args.lr)

    # Rebuild model from clean P2 for the actual training run.
    m, _ = build_model(Path(args.config), vocab_size=tok.vocab_size)
    m.load_state_dict(st["model_state_dict"])
    opt = torch.optim.AdamW(m.parameters(), lr=peak_lr, weight_decay=0.01)

    mixture = STAGE1_MIX if args.stage == "stage1" else STAGE2_MIX
    records = train_gated(model=m, optimizer=opt,
                                train_provider=train_provider,
                                val_provider=val_provider,
                                checkpoint_out=Path(args.out),
                                baseline_sigma=baseline_sigma,
                                mixture=mixture, l_native=args.l_native,
                                stage=args.stage, seed=args.seed,
                                lr_peak=peak_lr, lr_final=0.1 * peak_lr,
                                batch_size=args.batch_size)
    summary = {"stage": args.stage, "seed": args.seed, "lr_peak": peak_lr,
                  "checkpoints": [{"step": r.step, "tokens": r.tokens_covered,
                                        "val_loss": r.val_loss,
                                        "rel_imp": r.relative_improvement,
                                        "arch_ok": r.architecture_delta_zero,
                                        "stab_ok": r.native_stability_passed,
                                        "grad_ok": r.grad_path_ok,
                                        "candidate_release_id": r.manifest["candidate_release_id"]}
                                       for r in records]}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
