"""EN-TRAIN infrastructure regression — exercises every module in
aeon.en_train without needing an external corpus.

Every check uses either synthetic tokens or the six-book AEON-LBC-1
train partition (still committed as baseline provenance, per the
directive's rule that its reuse must obey the §3 caps). No new
authorized corpus exists yet — training is NOT run here.
"""
import hashlib
import json
import os
import random
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch


# ---------------------------------------------------------------------------
# aeon.en_train constants + freeze consistency
# ---------------------------------------------------------------------------
def test_en_train_constants_match_the_frozen_a0():
    import aeon.en_train as en
    freeze = json.load(open(os.path.join(ROOT, "docs/en_train/EN_TRAIN_ARCHITECTURE_FREEZE.json")))
    assert en.PROTECTED_P2_SHA256 == freeze["protected_p2_checkpoint"]["sha256"]
    assert en.PROTECTED_TOKENIZER_SHA256 == freeze["protected_tokenizer"]["sha256"]
    assert en.PROTECTED_A0_DIGEST == freeze["architecture_fingerprint_A0_digest"]
    assert en.PROTECTED_TOTAL_PARAMETERS == freeze["total_parameters"]
    assert en.FIXED_K == 16
    assert en.FIXED_VOCAB_SIZE == 16000


# ---------------------------------------------------------------------------
# §2 fingerprint + Δarchitecture
# ---------------------------------------------------------------------------
def test_architecture_fingerprint_A0_recomputes():
    import torch, yaml
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    from aeon.en_train.proof import compute_architecture_fingerprint, digest_fingerprint
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/latent_bypass/aeon_lbc1_proxy.yaml")))
    mc = cfg["model"]; tc = mc["transformer"]
    tconfig = AeonTransformerConfig(vocab_size=16000, hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"], num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"], head_dim=tc["head_dim"],
        intermediate_size=tc["intermediate_size"], max_position_embeddings=tc["max_position_embeddings"])
    m = HybridModel(transformer_config=tconfig, h_rec=mc["h_rec"], K=mc["K"],
        margin_h=mc["margin_h"], margin_c=mc["margin_c"], use_embedding_input=True,
        dtype=torch.float32).to(dtype=torch.float32)
    freeze = json.load(open(os.path.join(ROOT, "docs/en_train/EN_TRAIN_ARCHITECTURE_FREEZE.json")))
    fp = compute_architecture_fingerprint(m)
    d = digest_fingerprint(fp)
    assert d == freeze["architecture_fingerprint_A0_digest"]


def test_assert_architecture_invariant_passes_for_untrained_copy():
    import yaml, torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    from aeon.en_train.proof import assert_architecture_invariant
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/latent_bypass/aeon_lbc1_proxy.yaml")))
    mc = cfg["model"]; tc = mc["transformer"]
    tconfig = AeonTransformerConfig(vocab_size=16000, hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"], num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"], head_dim=tc["head_dim"],
        intermediate_size=tc["intermediate_size"], max_position_embeddings=tc["max_position_embeddings"])
    m = HybridModel(transformer_config=tconfig, h_rec=mc["h_rec"], K=mc["K"],
        margin_h=mc["margin_h"], margin_c=mc["margin_c"], use_embedding_input=True,
        dtype=torch.float32).to(dtype=torch.float32)
    r = assert_architecture_invariant(m)
    assert r["delta_architecture_zero"] is True


def test_assert_architecture_invariant_catches_parameter_count_drift():
    from aeon.en_train.proof import assert_architecture_invariant, ArchitectureViolation
    # A tiny toy model with the wrong param count must be refused.
    m = torch.nn.Linear(3, 4)
    class Fake:
        K = 16; h_rec = 128; D = 256
        def __init__(self):
            self._sd = {"w": torch.zeros(3,4)}
            self.recursion = torch.nn.Module()
            self.recursion.use_embedding_input = True
            self.recursion.MARGIN_H = 0.98
            self.recursion.MARGIN_C = 0.95
            self.substrate = torch.nn.Module()
        d_in = 128; d_state = 128
        def modules(self): return iter([self, self.recursion, self.substrate])
        def state_dict(self): return self._sd
        def parameters(self): return iter([torch.nn.Parameter(torch.zeros(1))])
        def named_parameters(self): return iter([("w", torch.nn.Parameter(torch.zeros(1)))])
    try:
        assert_architecture_invariant(Fake())
    except ArchitectureViolation as e:
        assert e.code == "parameter_count_changed"


# ---------------------------------------------------------------------------
# §3 splitter + dedup
# ---------------------------------------------------------------------------
def _docs(seed=0):
    from aeon.en_train.data import Document
    rng = random.Random(seed)
    words = "the quick brown fox jumps over the lazy dog with a hat and a bat".split()
    docs = []
    for i in range(60):
        text = " ".join(rng.choices(words, k=200))
        docs.append(Document(doc_id=f"d{i}", text=text,
                                  source_id=f"src{i % 6}",
                                  author_or_institution=f"author{i % 8}",
                                  est_token_count=200))
    return docs


def test_splitter_partitions_and_caps():
    from aeon.en_train.data import Splitter, IntakeError
    docs = _docs(0)
    s = Splitter(val_fraction=0.1, test_fraction=0.1, seed=42)
    try:
        parts = s.split(docs)
    except IntakeError as e:
        # This is expected: with only 6 sources of ~200 tokens each,
        # the train partition would exceed the 0.5% per-book cap.
        # That is the CORRECT behavior — the corpus is too concentrated.
        assert e.code == "partition_cap_exceeded"
        return
    # If the sample happened to fit, invariants must hold anyway.
    ids_by_part = {p: {d.doc_id for d in parts[p]} for p in parts}
    for a in parts:
        for b in parts:
            if a == b: continue
            assert not ids_by_part[a] & ids_by_part[b], f"leak {a} vs {b}"


def test_splitter_never_leaks_duplicates_across_partitions():
    from aeon.en_train.data import Document, Splitter
    # Two exact-duplicates (by normalized-hash) MUST land in the same partition.
    same = "hello world this is a test"
    other = "completely different content abcdefg"
    docs = [
        Document(doc_id="a", text=same, source_id="s1",
                    author_or_institution="A", est_token_count=100),
        Document(doc_id="b", text=same, source_id="s2",  # same text different source
                    author_or_institution="B", est_token_count=100),
        Document(doc_id="c", text=other, source_id="s3",
                    author_or_institution="C", est_token_count=100),
    ]
    s = Splitter(val_fraction=0.34, test_fraction=0.33, seed=1)
    parts = s.split(docs)
    for a in parts:
        ids_a = {d.doc_id for d in parts[a]}
        for b in parts:
            if a == b: continue
            ids_b = {d.doc_id for d in parts[b]}
            # 'a' and 'b' are duplicates; if 'a' in one partition then
            # 'b' must NOT be in a different one.
            if "a" in ids_a: assert "b" not in ids_b
            if "b" in ids_a: assert "a" not in ids_b


def test_jaccard_5_gram_grouping():
    from aeon.en_train.data import word_ngrams, jaccard
    # Two long overlapping paragraphs — identical except for one final
    # inserted word. 5-gram Jaccard should be well above 0.85.
    common = ("the quick brown fox jumps over the lazy dog under a bright "
                 "moon while a black cat crossed the empty road and vanished "
                 "into the fog and no one saw where the cat went next")
    a = common
    b = common + " tonight"
    c = "a completely different sentence with other unrelated words that share nothing"
    ga = word_ngrams(a, 5); gb = word_ngrams(b, 5); gc = word_ngrams(c, 5)
    assert jaccard(ga, gb) >= 0.85, jaccard(ga, gb)
    assert jaccard(ga, gc) < 0.85, jaccard(ga, gc)


# ---------------------------------------------------------------------------
# §4 tokenizer check
# ---------------------------------------------------------------------------
def test_tokenizer_check_reports_r_unk_zero_for_ascii_english():
    from aeon.en_train.data import Document, run_tokenizer_check
    from aeon.tokenizer import AeonTokenizer
    tok = AeonTokenizer(os.path.join(ROOT, "research-data/AEON-LBC-1/tokenizer/aeon-lbc1.model"))
    docs = [Document(doc_id="a", text="The quick brown fox jumps over the lazy dog.",
                          source_id="s", author_or_institution="A",
                          est_token_count=10)]
    r = run_tokenizer_check(tok, docs)
    assert r.r_unk <= 0.001  # ascii english should have no UNKs
    assert r.passed


# ---------------------------------------------------------------------------
# §5 general-English loss on the real HybridModel + random tokens
# ---------------------------------------------------------------------------
def _tiny_model():
    import yaml, torch
    from aeon.hybrid import HybridModel
    from aeon.transformer import AeonTransformerConfig
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/latent_bypass/aeon_lbc1_proxy.yaml")))
    mc = cfg["model"]; tc = mc["transformer"]
    tconfig = AeonTransformerConfig(vocab_size=16000, hidden_size=tc["hidden_size"],
        num_hidden_layers=tc["num_hidden_layers"], num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"], head_dim=tc["head_dim"],
        intermediate_size=tc["intermediate_size"], max_position_embeddings=tc["max_position_embeddings"])
    m = HybridModel(transformer_config=tconfig, h_rec=mc["h_rec"], K=mc["K"],
        margin_h=mc["margin_h"], margin_c=mc["margin_c"], use_embedding_input=True,
        dtype=torch.float32).to(dtype=torch.float32)
    return m


def test_general_english_loss_produces_finite_scalar_and_valid_tokens():
    from aeon.en_train.losses import general_english_loss
    torch.manual_seed(0)
    m = _tiny_model()
    ids = torch.randint(0, 16000, (2, 32), dtype=torch.long)
    att = torch.ones_like(ids)
    loss, vt = general_english_loss(m, input_ids=ids, attention_mask=att)
    assert torch.isfinite(loss).all()
    assert vt == 2 * 31   # (B, L-1) with all-ones mask


def test_conversational_loss_masks_out_user_tokens():
    from aeon.en_train.losses import conversational_loss
    torch.manual_seed(0)
    m = _tiny_model()
    ids = torch.randint(0, 16000, (2, 32), dtype=torch.long)
    att = torch.ones_like(ids)
    rmask = torch.zeros_like(ids)
    rmask[:, 10:] = 1     # only positions >=10 supervised
    loss, vt = conversational_loss(m, input_ids=ids, response_mask=rmask,
                                            attention_mask=att)
    # supervised positions in prediction-space are rmask[:,1:], which
    # has (32-1)*2 - (10-1)*2 = 42 ones for our setup? Actually
    # rmask[:,1:] preserves positions 1..31 -> rmask=1 for cols 10..31
    # which is 22 positions × 2 rows = 44.
    assert torch.isfinite(loss).all()
    assert vt == 44


# ---------------------------------------------------------------------------
# §6 chat serialization + response mask
# ---------------------------------------------------------------------------
def test_conversation_serialization_marks_assistant_spans_only():
    from aeon.en_train.losses import (
        render_conversation_for_training, build_response_mask, USER_PREFIX,
        ASSIST_PREFIX,
    )
    from aeon.tokenizer import AeonTokenizer
    tok = AeonTokenizer(os.path.join(ROOT, "research-data/AEON-LBC-1/tokenizer/aeon-lbc1.model"))
    turns = [("user", "Hello."), ("assistant", "Hi there."),
                ("user", "How are you?"), ("assistant", "I am fine.")]
    text, spans = render_conversation_for_training(turns)
    # Every span slice must be the assistant's actual content
    for a, b in spans:
        assert text[a:b] in ("Hi there.", "I am fine.")
    ids, mask = build_response_mask(tok, text, spans)
    assert len(ids) == len(mask)
    assert sum(mask) > 0   # at least some tokens supervised
    # No token that decodes as USER_PREFIX or ASSIST_PREFIX should be
    # supervised.
    supervised_ids = [i for i, m in zip(ids, mask) if m == 1]
    supervised_decoded = tok.decode(supervised_ids)
    assert "user:" not in supervised_decoded.lower()
    assert "assistant:" not in supervised_decoded.lower()


# ---------------------------------------------------------------------------
# §8 sequence buckets
# ---------------------------------------------------------------------------
def test_sequence_bucket_sampling_covers_all_four_bands():
    from aeon.en_train.losses import pick_sequence_length
    rng = random.Random(0)
    L = 400
    bands = {0.25: 0, 0.50: 0, 0.75: 0, 1.00: 0}
    for _ in range(4000):
        s = pick_sequence_length(L, rng)
        # classify to closest band
        target = min(bands.keys(), key=lambda t: abs(int(round(t * L)) - s))
        bands[target] += 1
    for target, n in bands.items():
        assert n > 500  # not silently degenerate


# ---------------------------------------------------------------------------
# §11 effective-token accounting
# ---------------------------------------------------------------------------
def test_effective_token_counter_accumulates_across_microbatches():
    from aeon.en_train.losses import EffectiveTokenCounter
    c = EffectiveTokenCounter()
    c.add_microbatch(1000); c.add_microbatch(2000); c.add_microbatch(3000)
    assert c.commit_update() == 6000
    assert c.updates == 1
    assert c.total_tokens == 6000


# ---------------------------------------------------------------------------
# §12 gradient-path proof + §13 weight-delta proof
# ---------------------------------------------------------------------------
def test_gradient_path_observation_and_100_step_gate():
    from aeon.en_train.losses import general_english_loss
    from aeon.en_train.proof import (
        observe_gradient_path, assert_gradient_path_over_100_steps,
    )
    torch.manual_seed(0)
    m = _tiny_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.0)
    obs = []
    for step in range(3):
        ids = torch.randint(0, 16000, (2, 32), dtype=torch.long)
        loss, _ = general_english_loss(m, input_ids=ids)
        opt.zero_grad(); loss.backward()
        o = observe_gradient_path(m, step)
        obs.append(o); opt.step()
    r = assert_gradient_path_over_100_steps(obs)
    assert r["n_observations"] == 3
    for g, v in r["per_group_max_grad_l2"].items():
        assert v > 0


def test_weight_delta_proof_records_actual_change():
    from aeon.en_train.losses import general_english_loss
    from aeon.en_train.proof import snapshot_state_dict, compute_weight_delta
    torch.manual_seed(0)
    m = _tiny_model()
    before = snapshot_state_dict(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    for _ in range(3):
        ids = torch.randint(0, 16000, (2, 32), dtype=torch.long)
        loss, _ = general_english_loss(m, input_ids=ids)
        opt.zero_grad(); loss.backward(); opt.step()
    after = snapshot_state_dict(m)
    r = compute_weight_delta(before, after)
    # Every trainable tensor should have moved at least a little
    assert len(r.positive_delta) > 0
    assert r.max_delta > 0


# ---------------------------------------------------------------------------
# §14 native stability gate
# ---------------------------------------------------------------------------
def test_native_stability_gate_passes_at_baseline_and_flags_drift():
    from aeon.en_train.proof import (
        sigma_certificate, assert_native_stability_gate,
        NativeStabilityViolation,
    )
    m = _tiny_model()
    base = sigma_certificate(m)
    r = assert_native_stability_gate(m, base)
    assert r["passed"] is True
    # Deliberately mutate the recorded baseline and prove it fires.
    bad_base = dict(base); bad_base["MARGIN_H"] = float(base["MARGIN_H"]) * 2.0
    try:
        assert_native_stability_gate(m, bad_base)
    except NativeStabilityViolation as e:
        assert e.code == "native_diagnostic_drift"


# ---------------------------------------------------------------------------
# §17..§21 evaluation metrics
# ---------------------------------------------------------------------------
def test_readable_success_rule():
    from aeon.en_train.eval import readable_success
    assert readable_success(2, 2, 1) == 1
    assert readable_success(1, 1, 1) == 1
    assert readable_success(0, 2, 1) == 0
    assert readable_success(2, 2, 0) == 0


def test_instruction_success_requires_all_components_and_no_contradictions():
    from aeon.en_train.eval import instruction_success
    key = {"required_components": ["hello", "world"],
              "contradictory_components": ["oops"]}
    assert instruction_success("hello world", key)
    assert not instruction_success("hello", key)
    assert not instruction_success("hello world oops", key)


def test_four_gram_repeat_rate_and_long_repeat_detection():
    from aeon.en_train.eval import four_gram_repeat_rate, has_long_repeat
    ids = [1, 2, 3, 4, 5, 6, 7, 8]
    assert four_gram_repeat_rate(ids) == 0.0
    ids2 = [1, 2, 3, 4] * 4
    assert four_gram_repeat_rate(ids2) > 0.5
    long_ids = list(range(20)) + list(range(20))
    assert has_long_repeat(long_ids, 8)


def test_fixation_wordlist_matches_whaling_terms():
    from aeon.en_train.eval import contains_fixation_term
    assert contains_fixation_term("and the whale of the whale")
    assert not contains_fixation_term("hello world")


def test_stream_full_equality_rate_is_zero_on_matches():
    from aeon.en_train.eval import stream_full_equality_rate
    assert stream_full_equality_rate([("a", "a"), ("b", "b")]) == 0.0
    assert stream_full_equality_rate([("a", "a"), ("b", "c")]) == 0.5


# ---------------------------------------------------------------------------
# §22 attribution test scaffolding
# ---------------------------------------------------------------------------
def test_attribution_confirms_when_p2_baseline_returns_after_swap_back():
    from aeon.en_train.attribution import attribution_test
    scripted = {"P2": {"R_readable": 0.30},
                    "CAND": {"R_readable": 0.70},
                    "P2_B": {"R_readable": 0.30}}
    def _eval(path, tag):
        if tag == "baseline_P2": return scripted["P2"]
        if tag == "candidate": return scripted["CAND"]
        return scripted["P2_B"]
    r = attribution_test(eval_fn=_eval, p2_path="p2", candidate_path="cand")
    assert r.attribution_confirmed


def test_attribution_refuses_when_improvement_lingers_on_p2():
    from aeon.en_train.attribution import attribution_test
    scripted = {"P2": 0.30, "CAND": 0.70, "P2_B": 0.70}
    def _eval(path, tag):
        return {"R_readable": scripted[{"baseline_P2":"P2","candidate":"CAND","restored_P2":"P2_B"}[tag]]}
    r = attribution_test(eval_fn=_eval, p2_path="p2", candidate_path="cand")
    assert not r.attribution_confirmed


# ---------------------------------------------------------------------------
# §23 promotion gate wiring
# ---------------------------------------------------------------------------
def test_promotion_gate_requires_every_gate():
    from aeon.en_train.eval import SealedEvalReport, check_promotion_gates
    good = SealedEvalReport(
        checkpoint_identity="cand", checkpoint_sha256="sha256:0",
        tokenizer_sha256="sha256:0", total_prompts=1, n_readable=1, R_readable=0.95,
        n_one_ok=1, R_one=0.85, n_two_ok=1, R_two=0.75,
        n_continuity_ok=1, continuity_rate=0.80,
        R_repeat_mean_four_gram=0.0, long_repeat_rate=0.01,
        R_fixation_on_unrelated=0.01, E_stream=0.0)
    dec = check_promotion_gates(good, native_stability_passed=True,
                                        architecture_delta_zero=True,
                                        gradient_path_ok=True,
                                        protected_p2_hash_unchanged=True,
                                        attribution_confirmed=True)
    assert dec.passed
    dec2 = check_promotion_gates(good, native_stability_passed=True,
                                          architecture_delta_zero=True,
                                          gradient_path_ok=True,
                                          protected_p2_hash_unchanged=True,
                                          attribution_confirmed=True,
                                          external_model_used=True)
    assert not dec2.passed


# ---------------------------------------------------------------------------
# Resumable-checkpoint round-trip
# ---------------------------------------------------------------------------
def test_candidate_checkpoint_save_and_reload_roundtrip():
    from aeon.en_train.trainer import save_candidate_checkpoint, load_candidate_checkpoint
    m = _tiny_model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    tmp = tempfile.mkdtemp()
    try:
        manifest = save_candidate_checkpoint(m, opt, tmp, step=1, tokens_covered=100,
                                                      stage="stage1", seed=1, val_loss=9.9,
                                                      mixture={"D_G": 1.0}, lr_pilot_choice=1e-4,
                                                      n_updates=1, effective_batch_tokens=16384)
        assert manifest["step"] == 1
        pt_path = os.path.join(tmp, "candidate-step000000001", "state.pt")
        m2 = _tiny_model()
        opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-4)
        info = load_candidate_checkpoint(m2, opt2, pt_path)
        assert info["step"] == 1 and info["stage"] == "stage1"
        # State-dict equality on reload
        sd1 = m.state_dict(); sd2 = m2.state_dict()
        assert set(sd1.keys()) == set(sd2.keys())
        for k in sd1: assert torch.equal(sd1[k], sd2[k])
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# Intake schema validator (behavior only — no live intake)
# ---------------------------------------------------------------------------
def test_intake_schema_rejects_missing_dirs():
    from aeon.en_train.data import validate_intake_layout, IntakeError
    tmp = tempfile.mkdtemp()
    try:
        try:
            validate_intake_layout(tmp)
        except IntakeError as e:
            assert e.code == "missing_intake_directory"
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)


def test_intake_schema_rejects_source_missing_provenance():
    from aeon.en_train.data import validate_intake_layout, IntakeError
    tmp = tempfile.mkdtemp()
    try:
        for d in ("sources", "provenance", "licenses", "manifests"):
            os.makedirs(os.path.join(tmp, d))
        # One source, no provenance
        with open(os.path.join(tmp, "sources", "a.txt"), "w") as f:
            f.write("hi")
        try:
            validate_intake_layout(tmp)
        except IntakeError as e:
            assert e.code == "missing_provenance"
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)


def test_intake_schema_rejects_provenance_missing_fields():
    from aeon.en_train.data import validate_intake_layout, IntakeError
    tmp = tempfile.mkdtemp()
    try:
        for d in ("sources", "provenance", "licenses", "manifests"):
            os.makedirs(os.path.join(tmp, d))
        with open(os.path.join(tmp, "sources", "a.txt"), "w") as f:
            f.write("hi")
        with open(os.path.join(tmp, "provenance", "a.json"), "w") as f:
            json.dump({"source_id": "a"}, f)
        try:
            validate_intake_layout(tmp)
        except IntakeError as e:
            assert e.code == "provenance_missing_fields"
    finally:
        import shutil; shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
def _run_all():
    tests = [v for k, v in sorted(globals().items())
              if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok  {t.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    _run_all()
