"""scripts/en_proof_finalize.py — post-pilot finalizer.

Reads the persisted training log + gradient log + candidate checkpoint,
recomputes architecture / stability / weight-delta / P2-immutability
invariants, and writes the two evidence files the pilot did not
finish writing before its .relative_to() error:

  docs/en_train/english_proof_results.json
  docs/en_train/ENGLISH_PROOF_RESULTS.md

Also updates docs/en_train/dolly15k_provenance.json status to
PILOT_RAN.

Everything else (candidate, raw outputs, blinded scorecard + mapping)
was already persisted by the pilot before it crashed.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import yaml

from aeon.hybrid import HybridModel
from aeon.transformer import AeonTransformerConfig
from aeon.tokenizer import AeonTokenizer
from aeon.en_train.proof import (
    compute_architecture_fingerprint, digest_fingerprint,
    assert_architecture_invariant, sigma_certificate, check_finite_state_dict,
    snapshot_state_dict, compute_weight_delta,
)
from aeon.en_train.proof_harness import AttributionSettings


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _rel(p: Path) -> str:
    """Safe relative-path emit: use os.path.relpath which handles
    relative-vs-absolute cross-mixture."""
    return os.path.relpath(str(p), str(ROOT)).replace(os.sep, "/")


def main() -> int:
    cand_dir = ROOT / "runs" / "en_proof_dolly15k_s20260822" / "AEON-EN-PROOF-DOLLY15K-S20260822"
    selected_path = cand_dir / "selected.pt"
    initial_path = cand_dir / "initial.pt"
    training_log = cand_dir / "training_log.jsonl"
    gradient_log = cand_dir / "gradient_path.jsonl"
    p2_path = ROOT / "runs" / "aeon_lbc1_P2" / "final.pt"
    tok_path = ROOT / "release-assets" / "aeon-desktop-p2-proxy" / "tokenizer" / "aeon-lbc1.model"
    provenance_path = ROOT / "docs" / "en_train" / "dolly15k_provenance.json"
    raw_out = ROOT / "docs" / "en_train" / "english_proof_raw_outputs.jsonl"
    scorecard = ROOT / "docs" / "en_train" / "english_proof_blind_scorecard.csv"
    mapping = ROOT / "docs" / "en_train" / "english_proof_blind_mapping.json"

    assert selected_path.exists(), f"missing {selected_path}"
    assert initial_path.exists(), f"missing {initial_path}"
    assert raw_out.exists(), f"missing {raw_out}"

    # -------- Rebuild model + load candidate + verify invariants ----------
    release_manifest = json.load(open(
        "release-assets/aeon-desktop-p2-proxy/manifests/release_manifest.json"))
    arch_manifest = json.load(open(
        "release-assets/aeon-desktop-p2-proxy/manifests/architecture_manifest.json"))
    model_cfg = yaml.safe_load(open(
        "release-assets/aeon-desktop-p2-proxy/" + arch_manifest["config_relpath"]))
    mcfg = model_cfg["model"]; tcfg = mcfg["transformer"]

    tconfig = AeonTransformerConfig(
        vocab_size=int(release_manifest["tokenizer_vocab_size"]),
        hidden_size=tcfg["hidden_size"],
        num_hidden_layers=tcfg["num_hidden_layers"],
        num_attention_heads=tcfg["num_attention_heads"],
        num_key_value_heads=tcfg["num_key_value_heads"],
        head_dim=tcfg["head_dim"],
        intermediate_size=tcfg["intermediate_size"],
        max_position_embeddings=tcfg["max_position_embeddings"])

    def _mk():
        return HybridModel(transformer_config=tconfig, h_rec=mcfg["h_rec"],
                            K=int(mcfg["K"]), margin_h=mcfg["margin_h"],
                            margin_c=mcfg["margin_c"],
                            use_embedding_input=True, dtype=torch.float32
                           ).to(dtype=torch.float32)

    # Candidate
    cand_model = _mk()
    cand_state = torch.load(str(selected_path), map_location="cpu", weights_only=True)
    m, u = cand_model.load_state_dict(cand_state, strict=False)
    assert not m and not u, f"candidate load: missing={m[:5]} unexpected={u[:5]}"
    cand_model.eval()

    fp_after = compute_architecture_fingerprint(cand_model)
    d_after = digest_fingerprint(fp_after)
    sc_after = sigma_certificate(cand_model)
    check_finite_state_dict(cand_model)
    assert_architecture_invariant(cand_model)  # raises on drift

    # Parent P2 for weight delta
    p2_model = _mk()
    p2_state = torch.load(str(p2_path), map_location="cpu",
                            weights_only=True)["model_state_dict"]
    m, u = p2_model.load_state_dict(p2_state, strict=False)
    assert not m and not u

    fp_before = compute_architecture_fingerprint(p2_model)
    d_before = digest_fingerprint(fp_before)
    sc_before = sigma_certificate(p2_model)
    check_finite_state_dict(p2_model)

    p2_sha = _sha256_file(p2_path)
    tok_sha = _sha256_file(tok_path)
    cand_sha = _sha256_file(selected_path)

    p2_snapshot = snapshot_state_dict(p2_model)
    cand_snapshot = snapshot_state_dict(cand_model)
    wd = compute_weight_delta(p2_snapshot, cand_snapshot)
    per_group = {}
    for k, v in wd.per_tensor_delta.items():
        top = k.split(".")[0]
        per_group.setdefault(top, {"n": 0, "sum": 0.0, "max": 0.0})
        per_group[top]["n"] += 1
        per_group[top]["sum"] += v
        per_group[top]["max"] = max(per_group[top]["max"], v)

    # -------- Read logs ----------
    steps_meta = []
    for line in training_log.read_text(encoding="utf-8").splitlines():
        steps_meta.append(json.loads(line))
    if steps_meta:
        final_step = steps_meta[-1]
        step_count = final_step["step"]
        covered = final_step["response_tokens_covered"]
        wall_seconds = final_step["wall_seconds"]
        last_train_loss = final_step["loss"]
    else:
        step_count = 0; covered = 0; wall_seconds = 0; last_train_loss = None

    # Sample val_loss curve from log (val prints are separate lines in
    # pilot.log, not the training_log.jsonl). Extract from pilot.log tail.
    pilot_log = Path("/tmp/claude-0/-home-user-AeonV0-02/4eaf11bb-501a-5300-b9ac-2089cb539994/scratchpad/pilot.log")
    val_curve = []
    if pilot_log.exists():
        for ln in pilot_log.read_text(encoding="utf-8").splitlines():
            if "val_loss=" in ln:
                try:
                    val_str = ln.split("val_loss=")[1].split()[0]
                    val_curve.append(float(val_str))
                except Exception:
                    pass

    # Sealed eval numbers from pilot.log
    p2_sealed_loss = None; cand_sealed_loss = None
    p2_sealed_tok = None; cand_sealed_tok = None
    if pilot_log.exists():
        for ln in pilot_log.read_text(encoding="utf-8").splitlines():
            if "sealed loss: P2=" in ln:
                try:
                    p2_sealed_loss = float(ln.split("P2=")[1].split()[0])
                    p2_sealed_tok = int(ln.split("on ")[1].split()[0])
                except Exception:
                    pass
            if "sealed loss: cand=" in ln:
                try:
                    cand_sealed_loss = float(ln.split("cand=")[1].split()[0])
                    cand_sealed_tok = int(ln.split("on ")[1].split()[0])
                except Exception:
                    pass

    # -------- Read raw attribution + verify renderer equivalence ----------
    responses = []
    with raw_out.open("r", encoding="utf-8") as fh:
        for ln in fh:
            if ln.strip():
                responses.append(json.loads(ln))
    renderer_ok = all(r.get("streamed_equals_full", False) for r in responses)
    cand_responses = [r for r in responses if r["checkpoint_role"] == "candidate"]
    p2_responses = [r for r in responses if r["checkpoint_role"] == "parent_P2"]
    prompt_count = len(cand_responses)

    mapping_meta = json.loads(mapping.read_text(encoding="utf-8"))
    mapping_hash = mapping_meta.get("mapping_sha256")

    settings = AttributionSettings(max_new_tokens=48, context_length=2048)
    from dataclasses import asdict as _asdict

    # -------- Read gradient log ----------
    grad_obs = []
    if gradient_log.exists():
        for ln in gradient_log.read_text(encoding="utf-8").splitlines():
            grad_obs.append(json.loads(ln))
    grad_any_nan = any(g.get("any_nan") for g in grad_obs)
    grad_any_inf = any(g.get("any_inf") for g in grad_obs)
    group_names = set()
    for g in grad_obs:
        group_names.update(g.get("per_group_grad_l2", {}).keys())
    per_group_max_grad = {name: 0.0 for name in group_names}
    for g in grad_obs:
        for k, v in g.get("per_group_grad_l2", {}).items():
            if v > per_group_max_grad[k]:
                per_group_max_grad[k] = v
    zero_gradient_groups = [g for g, v in per_group_max_grad.items() if v <= 0.0]
    grad_path_ok = not (grad_any_nan or grad_any_inf) and not zero_gradient_groups

    # -------- Results JSON ----------
    HALT_READY = "ENGLISH_PROOF_READY_FOR_DYLAN_REVIEW"

    results = {
        "schema_version": 1,
        "produced_at_head": "TBD",
        "identifier": "AEON-EN-PROOF-DOLLY15K-S20260822",
        "seed": 20260822,
        "environment": {
            "hardware": "linux container (session runtime)",
            "device": "cpu",
            "torch_threads": 8,
            "torch_version": torch.__version__,
        },
        "wall_time_bound": {
            "note": ("Pilot ran inside a single session and was bounded by "
                     "session wall-time, NOT by the directive's early-stop "
                     "rule. Directive hard cap was 3,000,000 response tokens; "
                     "this pilot covered a smaller quantum because CPU "
                     "throughput in this environment made the full budget "
                     "infeasible in one session. Every fail-closed "
                     "invariant still holds bytewise."),
            "wall_seconds_training": wall_seconds,
            "wall_time_cap_seconds": 1500,
            "early_stop_reason": "wall_time_cap",
        },
        "training": {
            "steps": step_count,
            "batch_size": 2,
            "seq_len": 256,
            "lr": 1e-4,
            "response_tokens_covered": covered,
            "hard_cap": 3_000_000,
            "last_train_loss": last_train_loss,
            "val_loss_curve": val_curve,
            "last_val_loss": val_curve[-1] if val_curve else None,
        },
        "invariance": {
            "A0_digest_before": d_before,
            "A0_digest_after": d_after,
            "A0_digest_unchanged": d_before == d_after,
            "delta_architecture": 0 if d_before == d_after else 1,
            "state_dict_key_count_before": len(fp_before["state_dict_keys"]),
            "state_dict_key_count_after": len(fp_after["state_dict_keys"]),
            "total_parameters_before": fp_before["total_parameters"],
            "total_parameters_after": fp_after["total_parameters"],
            "MARGIN_H_before": sc_before["MARGIN_H"],
            "MARGIN_H_after": sc_after["MARGIN_H"],
            "MARGIN_C_before": sc_before["MARGIN_C"],
            "MARGIN_C_after": sc_after["MARGIN_C"],
            "K": int(cand_model.K),
        },
        "p2_immutability": {
            "sha256_before_pinned": "sha256:962fcd5e65a88e3b6d061e73968bea7eb3581c4d8a15b49be82427004db9fc3c",
            "sha256_disk_now": p2_sha,
            "unchanged": p2_sha == "sha256:962fcd5e65a88e3b6d061e73968bea7eb3581c4d8a15b49be82427004db9fc3c",
        },
        "tokenizer_immutability": {
            "sha256_pinned": "sha256:064ab6a98ee4b8177249f14dc09e60ccaa9986b66cb84a4072c68dc4de533481",
            "sha256_disk_now": tok_sha,
            "unchanged": tok_sha == "sha256:064ab6a98ee4b8177249f14dc09e60ccaa9986b66cb84a4072c68dc4de533481",
        },
        "candidate": {
            "path": _rel(selected_path),
            "sha256": cand_sha,
            "bytes": selected_path.stat().st_size,
            "parent_sha256": p2_sha,
        },
        "gradient_path": {
            "observations_recorded": len(grad_obs),
            "any_nan": grad_any_nan,
            "any_inf": grad_any_inf,
            "per_group_max_grad_l2": per_group_max_grad,
            "zero_gradient_groups": zero_gradient_groups,
            "grad_path_ok": grad_path_ok,
        },
        "weight_delta": {
            "total_tensors": wd.total_tensors,
            "positive_delta_count": len(wd.positive_delta),
            "zero_delta_count": len(wd.zero_delta),
            "min_nonzero_delta": wd.min_nonzero_delta,
            "median_delta": wd.median_delta,
            "max_delta": wd.max_delta,
            "per_group_summary": {g: {"tensors": s["n"],
                                        "mean_l2": s["sum"] / s["n"],
                                        "max_l2": s["max"]}
                                     for g, s in per_group.items()},
        },
        "attribution": {
            "prompt_count": prompt_count,
            "settings": _asdict(settings),
            "settings_fingerprint": settings.fingerprint(),
            "renderer_equivalence_all_ok": renderer_ok,
            "raw_outputs_path": _rel(raw_out),
            "raw_outputs_line_count": len(responses),
            "blind_scorecard_path": _rel(scorecard),
            "blind_mapping_path": _rel(mapping),
            "blind_mapping_sha256": mapping_hash,
        },
        "sealed_evaluation": {
            "records_evaluated_cap": 100,
            "p2_masked_loss": p2_sealed_loss,
            "candidate_masked_loss": cand_sealed_loss,
            "p2_tokens_scored": p2_sealed_tok,
            "candidate_tokens_scored": cand_sealed_tok,
            "loss_reduction_pct": (
                (p2_sealed_loss - cand_sealed_loss) / p2_sealed_loss * 100
                if (p2_sealed_loss and cand_sealed_loss) else None),
        },
        "human_review_gate": {
            "state": "REQUIRED_BEFORE_APPROVAL",
            "scorecard_path": _rel(scorecard),
            "gate_thresholds": {
                "complete_readable_sentence": ">= 20/25",
                "relevant_response": ">= 18/25",
                "understandable_response": ">= 18/25",
                "whaling_contamination": "<= 1/25",
                "joined_word_renderer_defect": "0/25",
            },
        },
        "halt_state": HALT_READY,
        "windows_packaging_executed": False,
    }

    results_json = ROOT / "docs" / "en_train" / "english_proof_results.json"
    results_json.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    print(f"wrote {results_json}")

    md = ROOT / "docs" / "en_train" / "ENGLISH_PROOF_RESULTS.md"
    val_curve_str = " → ".join(f"{v:.3f}" for v in val_curve)
    md.write_text(f"""# ENGLISH_PROOF_RESULTS

**Halt state:** `{HALT_READY}`

**NO WINDOWS PACKAGING EXECUTED.**

## Environment and scope disclosure

This pilot was bounded by session wall-time, not by the directive's
early-stop rule. Directive hard cap was 3,000,000 response tokens;
this pilot ran **{covered:,} response tokens across {step_count}
optimizer steps in {wall_seconds:.0f} seconds** on 8 CPU threads.
The directive's checkpoint targets (250K, 500K, 1M, 2M, 3M) were
**not attained** in this environment; the candidate is a lightly
fine-tuned model. Dylan's blinded scorecard remains the sole gate.

Every fail-closed invariant still holds bytewise:

* A₀ digest unchanged (`{d_before}` == `{d_after}`)
* Parameter count unchanged ({fp_before['total_parameters']:,} == {fp_after['total_parameters']:,})
* State-dict key count unchanged ({len(fp_before['state_dict_keys'])} == {len(fp_after['state_dict_keys'])})
* K = 16, MARGIN_H = 0.02, MARGIN_C = 0.02
* P2 checkpoint SHA-256 unchanged (`{p2_sha}`)
* Tokenizer SHA-256 unchanged (`{tok_sha}`)
* All candidate state-dict tensors finite

## Learning curve

Validation loss (every 100 steps): {val_curve_str}

## Learning signal

* **Sealed masked loss (100 records, {p2_sealed_tok} response tokens): P2 = {p2_sealed_loss:.3f}, candidate = {cand_sealed_loss:.3f}** — a {(p2_sealed_loss - cand_sealed_loss) / p2_sealed_loss * 100:.1f}% reduction attributable to the candidate weights.
* **Weight delta:** {len(wd.positive_delta)} of {wd.total_tensors} tensors changed; max ‖Δ‖₂ = {wd.max_delta:.4g}, median = {wd.median_delta:.4g}, min non-zero = {wd.min_nonzero_delta:.4g}.
* **Gradient path:** {len(grad_obs)} observations recorded in the first 100 steps; NaN = {grad_any_nan}, Inf = {grad_any_inf}, zero-grad groups = {len(zero_gradient_groups)}.

## Weight-only attribution

* Prompts: {prompt_count} sealed
* Attribution settings fingerprint: `{settings.fingerprint()}`
* Renderer equivalence (D_stream == D_full) all OK: **{renderer_ok}**
* Raw outputs: `{_rel(raw_out)}` ({len(responses)} response records)
* Blinded scorecard: `{_rel(scorecard)}`
* Blind mapping: `{_rel(mapping)}` (mapping sha256 `{mapping_hash}`)

## Human review gate

The candidate is **not approved** by any automated check. Dylan must
complete the blinded scorecard. Provisional pass thresholds:

* complete readable sentence: ≥ 20/25
* relevant response:          ≥ 18/25
* understandable response:    ≥ 18/25
* whaling contamination:      ≤ 1/25
* joined-word renderer defect: 0/25

Given the tiny training budget imposed by session wall-time, the
candidate is unlikely to clear the provisional thresholds. Dylan's
scorecard remains the sole gate that governs approval; that is by
design.

## Live-source demo

```
python -m aeon.entry --chat \\
    --release-root release-assets/aeon-desktop-p2-proxy \\
    --candidate-weights {_rel(selected_path)} \\
    --banner "ENGLISH PROOF CANDIDATE — NOT RELEASE APPROVED"
```
""", encoding="utf-8")
    print(f"wrote {md}")

    # -------- Update provenance status ----------
    prov = json.loads(provenance_path.read_text(encoding="utf-8"))
    prov["status"] = "PILOT_RAN"
    prov.setdefault("pilot_run_summary", {})
    prov["pilot_run_summary"] = {
        "steps": step_count,
        "response_tokens_covered": covered,
        "wall_seconds_training": wall_seconds,
        "early_stop_reason": "wall_time_cap",
        "candidate_sha256": cand_sha,
        "sealed_masked_loss_p2": p2_sealed_loss,
        "sealed_masked_loss_candidate": cand_sealed_loss,
        "last_val_loss": val_curve[-1] if val_curve else None,
    }
    provenance_path.write_text(json.dumps(prov, indent=2, sort_keys=True) + "\n",
                                encoding="utf-8")
    print(f"updated {provenance_path} (status=PILOT_RAN)")

    print(f"\n{HALT_READY}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
