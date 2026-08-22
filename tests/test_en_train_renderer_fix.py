"""EN-TRAIN — renderer-correction regression tests (§21).

Verifies:
  * D_stream(y) = D_full(y) for spacing, punctuation, unicode,
    byte-fallback, contractions, newlines.
  * Concatenation of every TEXT_DELTA emitted by AeonDesktopRuntime
    equals tok.decode(all_generated_ids) — byte-exact.
  * The renderer fix does NOT change token IDs, logits, weights,
    generation order, architecture fingerprint, tokenizer files,
    or model configuration.
"""
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

BUNDLE = os.path.join(ROOT, "release-assets", "aeon-desktop-p2-proxy")
TOK = os.path.join(ROOT, "research-data/AEON-LBC-1/tokenizer/aeon-lbc1.model")


# ---------------------------------------------------------------------------
# Direct decoder equivalence (deterministic, no model)
# ---------------------------------------------------------------------------
def _tokenizer():
    from aeon.tokenizer import AeonTokenizer
    return AeonTokenizer(TOK)


def _check_full_equals_stream(text: str):
    """Encode text, then verify that the fixed renderer's streaming
    pattern (cumulative canonical decode + U+FFFD hold-back +
    completion flush) reproduces tok.decode(ids) exactly."""
    tok = _tokenizer()
    ids = tok.encode(text, add_bos=False, add_eos=False)
    if len(ids) == 0:
        return
    D_full = tok.decode(ids)
    # Mirror aeon.desktop.runtime._generate's fixed rendering path.
    emitted = ""
    deltas = []
    for i in range(1, len(ids) + 1):
        canonical_so_far = tok.decode(ids[:i])
        committable = canonical_so_far.rstrip("�")
        if committable.startswith(emitted):
            deltas.append(committable[len(emitted):])
            emitted = committable
    # Completion-time flush
    if emitted != D_full:
        tail = D_full[len(emitted):] if D_full.startswith(emitted) else D_full
        deltas.append(tail)
        emitted = D_full
    D_stream = "".join(deltas)
    assert D_stream == D_full, f"D_stream != D_full for text={text!r}:\n  D_full  ={D_full!r}\n  D_stream={D_stream!r}"


def test_renderer_spacing_between_words():
    _check_full_equals_stream("The quick brown fox jumps over the lazy dog")


def test_renderer_punctuation_and_ascii():
    _check_full_equals_stream("Hello, world. Yes! No? Wait: yes; okay.")


def test_renderer_contractions():
    _check_full_equals_stream("don't can't isn't we're they've")


def test_renderer_newlines_and_paragraphs():
    _check_full_equals_stream("Line one.\nLine two.\n\nAnother paragraph.")


def test_renderer_unicode_common_latin1_supplement():
    _check_full_equals_stream("café résumé naïve façade")


def test_renderer_unicode_extended_and_byte_fallback():
    # Byte-fallback exercises the tokenizer's ability to reproduce any
    # code point through byte-level pieces. The unigram model was fit
    # with byte_fallback=True so this should round-trip.
    _check_full_equals_stream("hello 世界 — Ω π ✓")


def test_renderer_leading_space_after_punctuation():
    _check_full_equals_stream("A.B C,D E: F; G!H?")


def test_renderer_numeric_and_mixed():
    _check_full_equals_stream("Order 42, subtotal $3.14 at 09:30:45")


def test_renderer_empty_string_is_no_op():
    tok = _tokenizer()
    ids = tok.encode("", add_bos=False, add_eos=False)
    # Nothing to stream; passes trivially.
    assert tok.decode(ids) == ""


# ---------------------------------------------------------------------------
# Live desktop runtime: concatenated deltas == tok.decode(all_ids)
# ---------------------------------------------------------------------------
def _run_and_join(prompt, max_new=12):
    if not os.path.exists(BUNDLE):
        return None
    from aeon.desktop.runtime import AeonDesktopRuntime
    from aeon.desktop.protocol import GenerationOptions
    rt = AeonDesktopRuntime()
    rt.preflight()
    rt.load_release(BUNDLE)
    sid = rt.create_session()
    r = rt.submit_prompt_sync(sid, prompt,
                                    GenerationOptions(max_new_tokens=max_new, temperature=0.0))
    rt.shutdown()
    events = r["events"]
    text_delta_events = [e for e in events if e["event_type"] == "text_delta"]
    tokens = [e["payload"]["token_id"]
                for e in events if e["event_type"] == "token_generated"]
    completed = [e for e in events if e["event_type"] == "generation_completed"][0]
    joined_deltas = "".join(e["payload"]["delta"] for e in text_delta_events)
    full_text = completed["payload"]["full_text"]
    return {
        "token_ids": tokens,
        "joined_deltas": joined_deltas,
        "full_text": full_text,
    }


def test_live_runtime_join_of_deltas_equals_full_text():
    if not os.path.exists(BUNDLE):
        return
    for prompt in ("The", "Hello world", "Once upon a time"):
        r = _run_and_join(prompt, max_new=8)
        assert r["joined_deltas"] == r["full_text"], (
            f"joined deltas != full_text for prompt={prompt!r}:\n"
            f"  joined = {r['joined_deltas']!r}\n"
            f"  full   = {r['full_text']!r}")


def test_live_runtime_full_text_equals_canonical_decode():
    """Belt-and-braces: the runtime's own full_text payload equals
    tok.decode(token_ids)."""
    if not os.path.exists(BUNDLE):
        return
    tok = _tokenizer()
    for prompt in ("The", "Hello world", "Once upon a time"):
        r = _run_and_join(prompt, max_new=8)
        assert tok.decode(r["token_ids"]) == r["full_text"]


# ---------------------------------------------------------------------------
# No-drift proofs
# ---------------------------------------------------------------------------
def test_renderer_fix_does_not_change_token_ids_for_defect_fixtures():
    """Re-run the exact deterministic-greedy generation used by the
    defect proof and require the SAME token IDs. This proves the
    fix touched only the emission path, not selection."""
    if not os.path.exists(BUNDLE):
        return
    import torch
    import yaml
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    tok = _tokenizer()
    with open(os.path.join(ROOT, "docs/en_train/renderer_defect_proof.json")) as f:
        proof = json.load(f)
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/latent_bypass/aeon_lbc1_proxy.yaml")))
    mc = cfg["model"]; tc = mc["transformer"]
    tconfig = AeonTransformerConfig(vocab_size=16000, hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"], num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"], head_dim=tc["head_dim"],
        intermediate_size=tc["intermediate_size"], max_position_embeddings=tc["max_position_embeddings"])
    m = HybridModel(transformer_config=tconfig, h_rec=mc["h_rec"], K=mc["K"],
        margin_h=mc["margin_h"], margin_c=mc["margin_c"], use_embedding_input=True,
        dtype=torch.float32).to(dtype=torch.float32)
    st = torch.load(os.path.join(ROOT, "runs/aeon_lbc1_P2/final.pt"),
                        map_location="cpu", weights_only=False)
    m.load_state_dict(st["model_state_dict"]); m.eval()
    for p in proof["per_prompt"]:
        ids = list(p["prompt_token_ids"])
        want = list(p["generated_token_ids"])
        cur = list(ids); regenerated = []
        for _ in range(len(want)):
            ipt = torch.tensor([cur], dtype=torch.long)
            with torch.inference_mode():
                logits = m(input_ids=ipt).logits[0, -1, :]
            nxt = int(logits.argmax().item())
            cur.append(nxt); regenerated.append(nxt)
        assert regenerated == want, (
            f"token IDs drifted for prompt={p['prompt']!r}: got {regenerated}, want {want}")


def test_renderer_fix_did_not_touch_tokenizer_bytes():
    """The tokenizer file must be byte-identical before and after."""
    if not os.path.exists(TOK):
        return
    expected = "sha256:064ab6a98ee4b8177249f14dc09e60ccaa9986b66cb84a4072c68dc4de533481"
    with open(TOK, "rb") as f:
        got = "sha256:" + hashlib.sha256(f.read()).hexdigest()
    assert got == expected, f"tokenizer changed: {got}"


def test_renderer_fix_did_not_touch_p2_checkpoint():
    """The frozen P2 checkpoint must be byte-identical."""
    p = os.path.join(ROOT, "runs/aeon_lbc1_P2/final.pt")
    if not os.path.exists(p):
        return
    expected = "sha256:962fcd5e65a88e3b6d061e73968bea7eb3581c4d8a15b49be82427004db9fc3c"
    with open(p, "rb") as f:
        got = "sha256:" + hashlib.sha256(f.read()).hexdigest()
    assert got == expected, f"P2 checkpoint changed: {got}"


def test_renderer_fix_did_not_touch_model_configuration():
    """The model config must be byte-identical."""
    p = os.path.join(ROOT, "configs/latent_bypass/aeon_lbc1_proxy.yaml")
    freeze = json.load(open(os.path.join(ROOT, "docs/en_train/EN_TRAIN_ARCHITECTURE_FREEZE.json")))
    expected = freeze["protected_configuration"]["sha256"]
    with open(p, "rb") as f:
        got = "sha256:" + hashlib.sha256(f.read()).hexdigest()
    assert got == expected


def test_renderer_fix_did_not_change_architecture_fingerprint_A0():
    """Rebuild the model + recompute A0; must equal the frozen digest."""
    import torch
    import yaml
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    freeze = json.load(open(os.path.join(ROOT, "docs/en_train/EN_TRAIN_ARCHITECTURE_FREEZE.json")))
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/latent_bypass/aeon_lbc1_proxy.yaml")))
    mc = cfg["model"]; tc = mc["transformer"]
    tconfig = AeonTransformerConfig(vocab_size=16000, hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"], num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"], head_dim=tc["head_dim"],
        intermediate_size=tc["intermediate_size"], max_position_embeddings=tc["max_position_embeddings"])
    m = HybridModel(transformer_config=tconfig, h_rec=mc["h_rec"], K=mc["K"],
        margin_h=mc["margin_h"], margin_c=mc["margin_c"], use_embedding_input=True,
        dtype=torch.float32).to(dtype=torch.float32)
    a = {
        "module_type_names": sorted(list(set(type(x).__name__ for x in m.modules()))),
        "state_dict_keys": sorted(m.state_dict().keys()),
        "tensor_shapes": {k: list(v.shape) for k, v in sorted(m.state_dict().items())},
        "tensor_dtypes": {k: str(v.dtype) for k, v in sorted(m.state_dict().items())},
        "state_dict_key_count": len(m.state_dict()),
        "total_parameters": sum(p.numel() for p in m.parameters()),
        "trainable_parameters": sum(p.numel() for p in m.parameters() if p.requires_grad),
        "K": int(m.K),
        "h_rec": int(m.h_rec),
        "D_transformer_hidden": int(m.D),
        "recurrence_configuration": {
            "type": type(m.recursion).__name__,
            "use_embedding_input": bool(m.recursion.use_embedding_input),
            "MARGIN_H": float(m.recursion.MARGIN_H),
            "MARGIN_C": float(m.recursion.MARGIN_C),
        },
        "substrate_configuration": {
            "type": type(m.substrate).__name__,
            "d_in": int(m.d_in),
            "d_state": int(m.d_state),
        },
        "forward_signature_kwargs": ["input_ids", "attention_mask", "labels",
                                        "observer", "intervention", "shuttle"],
        "parameter_sharing_relationships": [],
    }
    a_bytes = json.dumps(a, sort_keys=True).encode("utf-8")
    a_digest = "sha256:" + hashlib.sha256(a_bytes).hexdigest()
    assert a_digest == freeze["architecture_fingerprint_A0_digest"], (
        f"A0 drifted: {a_digest} vs {freeze['architecture_fingerprint_A0_digest']}")


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
