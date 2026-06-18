"""
Per-layer byte-identity diagnostic — localizes where Aeon's transformer diverges
from HF Qwen2 in bf16.

Runs both models on identical input with forward hooks capturing every
checkpoint (embeddings, each decoder layer's sub-outputs, each layer output,
final norm, logits) and reports max|Δ| (computed in fp32) at each. The first
checkpoint whose diff jumps past the threshold is where divergence is introduced
— a sharp jump at one op = structural bug there; smooth growth = accumulation.

Sub-layer checkpoints per layer (both models share these attribute names):
  input_ln  -> self_attn (o_proj out)  -> post_ln  -> mlp  -> layer (residual out)

`transformers` is imported ONLY here (test-time). HF uses eager attention to
match Aeon. REQUIRES torch + transformers + AEON_R1_DIR (local R1 dir). dtype via
AEON_DIFF_DTYPE (default bfloat16). Skips cleanly when unavailable.

Run:  AEON_R1_DIR=/path AEON_DIFF_DTYPE=bfloat16 python tests/test_layer_diff.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

THRESHOLD = 1e-2   # "meaningful" divergence for bf16 localization


def _prereqs():
    d = os.environ.get("AEON_R1_DIR")
    if not d or not os.path.isdir(d):
        return None, "AEON_R1_DIR not set / not a directory"
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except Exception as e:
        return None, f"torch/transformers unavailable: {e}"
    return d, None


def _capture(store, name):
    def hook(_mod, _inp, out):
        t = out[0] if isinstance(out, tuple) else out
        store[name] = t.detach().float()
    return hook


def run_layer_diff():
    import torch
    from transformers import AutoModelForCausalLM
    from aeon.transformer import AeonQwen2Model, config_from_pretrained, load_r1_weights

    d, why = _prereqs()
    if d is None:
        print(f"  [skip] {why}")
        return
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[
        os.environ.get("AEON_DIFF_DTYPE", "bfloat16")]

    cfg = config_from_pretrained(d)
    aeon = AeonQwen2Model(cfg).to(dtype).eval()
    load_r1_weights(aeon, d)
    hf = AutoModelForCausalLM.from_pretrained(
        d, torch_dtype=dtype, attn_implementation="eager").eval()

    a_store, h_store = {}, {}
    hooks = []

    # embeddings
    hooks.append(aeon.embed_tokens.register_forward_hook(_capture(a_store, "embed")))
    hooks.append(hf.model.embed_tokens.register_forward_hook(_capture(h_store, "embed")))
    # layers + sublayers
    for i, (al, hl) in enumerate(zip(aeon.layers, hf.model.layers)):
        for sub in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
            key = f"layer{i:02d}.{sub}"
            hooks.append(getattr(al, sub).register_forward_hook(_capture(a_store, key)))
            hooks.append(getattr(hl, sub).register_forward_hook(_capture(h_store, key)))
        hooks.append(al.register_forward_hook(_capture(a_store, f"layer{i:02d}")))
        hooks.append(hl.register_forward_hook(_capture(h_store, f"layer{i:02d}")))
    # final norm
    hooks.append(aeon.norm.register_forward_hook(_capture(a_store, "final_norm")))
    hooks.append(hf.model.norm.register_forward_hook(_capture(h_store, "final_norm")))

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, 32))
    with torch.no_grad():
        a_logits = aeon.logits(aeon.forward_hidden(input_ids=ids)).float()
        h_logits = hf(input_ids=ids).logits.float()
    for hk in hooks:
        hk.remove()

    a_store["logits"] = a_logits
    h_store["logits"] = h_logits

    # ordered checkpoint list
    order = ["embed"]
    for i in range(len(aeon.layers)):
        order += [f"layer{i:02d}.input_layernorm", f"layer{i:02d}.self_attn",
                  f"layer{i:02d}.post_attention_layernorm", f"layer{i:02d}.mlp",
                  f"layer{i:02d}"]
    order += ["final_norm", "logits"]

    print(f"\nPer-checkpoint max|Δ| (dtype={dtype}, threshold={THRESHOLD}):")
    first = None
    for name in order:
        if name not in a_store or name not in h_store:
            continue
        diff = (a_store[name] - h_store[name]).abs().max().item()
        flag = ""
        if diff >= THRESHOLD and first is None:
            first = name
            flag = "  <-- FIRST meaningful divergence"
        print(f"  {name:34s} {diff:.3e}{flag}")
    if first is None:
        print(f"\nNo checkpoint exceeded {THRESHOLD}.")
    else:
        print(f"\nFirst meaningful divergence at: {first}")


def test_layer_diff():
    run_layer_diff()


if __name__ == "__main__":
    run_layer_diff()
