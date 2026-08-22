"""aeon.en_train.eval — deterministic sealed-eval runner + metrics.

Implements §15..§21 without invoking any external / LLM judge. Metrics
that require human judgment (grammar, relevance, completeness) are
scored from a scoring-key file supplied by the operator; the runner
never scores model output using another model.

`E_stream` is verified against the FIXED renderer at
`aeon.desktop.runtime.AeonDesktopRuntime._generate` (post-EN-TRAIN-1).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from . import (
    GATE_CONTINUITY, GATE_E_STREAM, GATE_LONG_REPEAT_MAX,
    GATE_R_FIXATION_MAX, GATE_R_ONE, GATE_R_READABLE, GATE_R_TWO,
)


# ---------------------------------------------------------------------------
# Sealed evaluation set
# ---------------------------------------------------------------------------
REQUIRED_SEALED_EVAL_CATEGORIES = (
    "greetings",              # ≥ 50
    "factual_from_context",   # ≥ 50
    "instruction_one_part",   # ≥ 50
    "instruction_two_part",   # ≥ 50
    "uncertainty",            # ≥ 40
    "continuity_two_turn",    # ≥ 40
    "identity",               # ≥ 20
)
REQUIRED_SEALED_EVAL_MINIMUMS = {
    "greetings": 50, "factual_from_context": 50,
    "instruction_one_part": 50, "instruction_two_part": 50,
    "uncertainty": 40, "continuity_two_turn": 40, "identity": 20,
}


@dataclass(frozen=True)
class SealedPrompt:
    prompt_id: str
    category: str
    prompt_text: str
    scoring_key: Dict[str, Any]  # human-authored deterministic rules
    context_turns: Optional[List[Dict[str, str]]] = None  # for continuity_two_turn


def load_sealed_eval(path: Path) -> List[SealedPrompt]:
    """Load a jsonl file of SealedPrompt records. Verifies category
    minimums per §16."""
    prompts: List[SealedPrompt] = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        prompts.append(SealedPrompt(
            prompt_id=r["prompt_id"],
            category=r["category"],
            prompt_text=r["prompt_text"],
            scoring_key=r.get("scoring_key", {}),
            context_turns=r.get("context_turns"),
        ))
    counts: Dict[str, int] = {}
    for p in prompts: counts[p.category] = counts.get(p.category, 0) + 1
    for cat, minimum in REQUIRED_SEALED_EVAL_MINIMUMS.items():
        if counts.get(cat, 0) < minimum:
            raise RuntimeError(f"sealed_eval_undersized: {cat}={counts.get(cat,0)}<{minimum}")
    return prompts


# ---------------------------------------------------------------------------
# §17 readability score
# ---------------------------------------------------------------------------
def readable_success(grammar: int, relevance: int, completeness: int) -> int:
    """§17: Readable(r) = 1 iff grammar>=1, relevance>=1, completeness==1."""
    if grammar >= 1 and relevance >= 1 and completeness == 1:
        return 1
    return 0


def score_readability_from_key(response: str,
                                       key: Dict[str, Any]) -> Tuple[int, int, int]:
    """Return (grammar, relevance, completeness) using deterministic
    rules from the scoring key. Every rule is either a regex or a
    literal-token check — no model-based judgment."""
    # required_contains: list of case-insensitive substrings that must
    # ALL appear for relevance to count as full (2).
    # required_regex: list of regex patterns that must ALL match.
    # forbidden_contains: list of substrings that must NOT appear.
    # min_chars / max_chars: bracket for completeness.
    r = response
    r_lower = r.lower()

    # Completeness: character-length window from the key.
    minc = int(key.get("min_chars", 1))
    maxc = int(key.get("max_chars", 100_000))
    completeness = 1 if minc <= len(r.strip()) <= maxc else 0

    # Grammar: 2 if no forbidden token, ≥ minc chars, and at least
    #   one sentence-ending punctuation. 1 if partial. 0 if empty.
    if not r.strip():
        grammar = 0
    elif any(bad.lower() in r_lower for bad in key.get("forbidden_contains", [])):
        grammar = 0
    elif re.search(r"[.!?]", r):
        grammar = 2
    else:
        grammar = 1

    # Relevance: 2 if ALL required_contains + ALL required_regex match,
    # 1 if at least one matches, 0 otherwise.
    contains = key.get("required_contains", [])
    regexes = [re.compile(p, re.IGNORECASE) for p in key.get("required_regex", [])]
    match_contains = sum(1 for s in contains if s.lower() in r_lower)
    match_regexes = sum(1 for pat in regexes if pat.search(r))
    total = len(contains) + len(regexes)
    hits = match_contains + match_regexes
    if total == 0:
        relevance = 1
    elif hits == total:
        relevance = 2
    elif hits > 0:
        relevance = 1
    else:
        relevance = 0
    return grammar, relevance, completeness


# ---------------------------------------------------------------------------
# §18 instruction-following rate
# ---------------------------------------------------------------------------
def instruction_success(response: str, key: Dict[str, Any]) -> bool:
    """§18: a response succeeds only if every explicitly requested
    component is present AND no contradictory instruction was
    followed."""
    for req in key.get("required_components", []):
        if req.lower() not in response.lower():
            return False
    for bad in key.get("contradictory_components", []):
        if bad.lower() in response.lower():
            return False
    return True


# ---------------------------------------------------------------------------
# §19 repetition rate
# ---------------------------------------------------------------------------
def four_gram_repeat_rate(token_ids: Sequence[int]) -> float:
    if len(token_ids) < 4: return 0.0
    grams = [tuple(token_ids[i:i + 4]) for i in range(len(token_ids) - 3)]
    return 1 - len(set(grams)) / max(1, len(grams))


def has_long_repeat(token_ids: Sequence[int], run: int = 8) -> bool:
    """True if there is a run of `run` consecutive tokens that appears
    at least twice (§19 "repeated phrase of eight or more consecutive
    tokens")."""
    if len(token_ids) < 2 * run: return False
    seen = set()
    for i in range(len(token_ids) - run + 1):
        g = tuple(token_ids[i:i + run])
        if g in seen: return True
        seen.add(g)
    return False


# ---------------------------------------------------------------------------
# §20 literary fixation
# ---------------------------------------------------------------------------
DEFAULT_LITERARY_FIXATION_WORDLIST = (
    "pequod", "queequeg", "ishmael", "ahab",
    "whale", "whaling", "whalebone", "harpoon",
)


def contains_fixation_term(response: str,
                                  wordlist: Sequence[str] = DEFAULT_LITERARY_FIXATION_WORDLIST
                                  ) -> bool:
    r = response.lower()
    for w in wordlist:
        if re.search(rf"\b{re.escape(w)}\b", r):
            return True
    return False


# ---------------------------------------------------------------------------
# §21 renderer integrity
# ---------------------------------------------------------------------------
def stream_full_equality_rate(runs: Sequence[Tuple[str, str]]) -> float:
    """`runs` is a list of (D_stream, D_full) pairs. Returns the fraction
    that MISMATCH (E_stream)."""
    if not runs: return 0.0
    return sum(1 for s, f in runs if s != f) / len(runs)


# ---------------------------------------------------------------------------
# End-to-end sealed-eval scoring
# ---------------------------------------------------------------------------
@dataclass
class SealedEvalReport:
    checkpoint_identity: str
    checkpoint_sha256: str
    tokenizer_sha256: str
    total_prompts: int
    n_readable: int
    R_readable: float
    n_one_ok: int
    R_one: float
    n_two_ok: int
    R_two: float
    n_continuity_ok: int
    continuity_rate: float
    R_repeat_mean_four_gram: float
    long_repeat_rate: float
    R_fixation_on_unrelated: float
    E_stream: float
    per_prompt: List[Dict[str, Any]] = field(default_factory=list)


def evaluate_checkpoint_deterministic(*,
                                              generate_fn,
                                              prompts: Sequence[SealedPrompt],
                                              checkpoint_identity: str,
                                              checkpoint_sha256: str,
                                              tokenizer_sha256: str,
                                              literary_wordlist: Sequence[str]
                                                = DEFAULT_LITERARY_FIXATION_WORDLIST,
                                              ) -> SealedEvalReport:
    """`generate_fn(prompt: str) -> dict{full_text, token_ids, joined_deltas}`
    must be a fully deterministic greedy generation (§15). The runner
    itself does not add sampling or prompt engineering."""
    n_readable = n_one_ok = n_two_ok = n_cont_ok = 0
    n_one = n_two = n_cont = 0
    n_unrelated = fix_hits = 0
    four_gram_sum = long_hits = 0
    stream_full_pairs: List[Tuple[str, str]] = []
    per: List[Dict[str, Any]] = []
    for p in prompts:
        r = generate_fn(p.prompt_text if p.category != "continuity_two_turn"
                             else p.prompt_text,
                            context_turns=p.context_turns)
        response = r["full_text"]
        token_ids = r["token_ids"]
        joined = r.get("joined_deltas", response)
        # §21
        stream_full_pairs.append((joined, response))
        # §17
        g, rel, comp = score_readability_from_key(response, p.scoring_key)
        rs = readable_success(g, rel, comp)
        if rs: n_readable += 1
        # §18
        if p.category == "instruction_one_part":
            n_one += 1
            if instruction_success(response, p.scoring_key):
                n_one_ok += 1
        elif p.category == "instruction_two_part":
            n_two += 1
            if instruction_success(response, p.scoring_key):
                n_two_ok += 1
        # §16 two-turn continuity
        if p.category == "continuity_two_turn":
            n_cont += 1
            if instruction_success(response, p.scoring_key) and rs:
                n_cont_ok += 1
        # §19
        four_gram_sum += four_gram_repeat_rate(token_ids)
        if has_long_repeat(token_ids, 8): long_hits += 1
        # §20 (only on unrelated prompts — anything not in the whale
        # / literary category counts)
        if p.category not in ("factual_from_context",) and \
                not any(w in p.prompt_text.lower() for w in ("whale", "queequeg", "pequod", "ishmael")):
            n_unrelated += 1
            if contains_fixation_term(response, literary_wordlist): fix_hits += 1
        per.append({
            "prompt_id": p.prompt_id,
            "category": p.category,
            "grammar": g, "relevance": rel, "completeness": comp,
            "readable": rs,
            "response_len_chars": len(response),
            "token_count": len(token_ids),
            "four_gram_repeat_rate": four_gram_repeat_rate(token_ids),
            "has_long_repeat": has_long_repeat(token_ids, 8),
            "stream_equals_full": joined == response,
        })
    total = len(prompts)
    return SealedEvalReport(
        checkpoint_identity=checkpoint_identity,
        checkpoint_sha256=checkpoint_sha256,
        tokenizer_sha256=tokenizer_sha256,
        total_prompts=total,
        n_readable=n_readable,
        R_readable=(n_readable / total) if total else 0.0,
        n_one_ok=n_one_ok,
        R_one=(n_one_ok / n_one) if n_one else 0.0,
        n_two_ok=n_two_ok,
        R_two=(n_two_ok / n_two) if n_two else 0.0,
        n_continuity_ok=n_cont_ok,
        continuity_rate=(n_cont_ok / n_cont) if n_cont else 0.0,
        R_repeat_mean_four_gram=(four_gram_sum / total) if total else 0.0,
        long_repeat_rate=(long_hits / total) if total else 0.0,
        R_fixation_on_unrelated=(fix_hits / n_unrelated) if n_unrelated else 0.0,
        E_stream=stream_full_equality_rate(stream_full_pairs),
        per_prompt=per,
    )


# ---------------------------------------------------------------------------
# §23 promotion decision
# ---------------------------------------------------------------------------
@dataclass
class PromotionDecision:
    passed: bool
    reasons: List[str]
    gates: Dict[str, Any]


def check_promotion_gates(report: SealedEvalReport,
                                 native_stability_passed: bool,
                                 architecture_delta_zero: bool,
                                 gradient_path_ok: bool,
                                 protected_p2_hash_unchanged: bool,
                                 attribution_confirmed: bool,
                                 external_model_used: bool = False,
                                 rewriting_system_used: bool = False,
                                 retrieval_answer_used: bool = False,
                                 canned_response_used: bool = False,
                                 ) -> PromotionDecision:
    reasons: List[str] = []
    def _need(cond: bool, msg: str) -> None:
        if not cond: reasons.append(msg)

    _need(architecture_delta_zero, "Δarchitecture ≠ 0")
    _need(protected_p2_hash_unchanged, "protected P2 hash changed")
    _need(report.E_stream == GATE_E_STREAM, f"E_stream={report.E_stream} != 0")
    _need(report.R_readable >= GATE_R_READABLE, f"R_readable={report.R_readable:.3f} < {GATE_R_READABLE}")
    _need(report.R_one >= GATE_R_ONE, f"R_one={report.R_one:.3f} < {GATE_R_ONE}")
    _need(report.R_two >= GATE_R_TWO, f"R_two={report.R_two:.3f} < {GATE_R_TWO}")
    _need(report.continuity_rate >= GATE_CONTINUITY, f"continuity={report.continuity_rate:.3f} < {GATE_CONTINUITY}")
    _need(report.R_fixation_on_unrelated <= GATE_R_FIXATION_MAX,
             f"R_fixation={report.R_fixation_on_unrelated:.3f} > {GATE_R_FIXATION_MAX}")
    _need(report.long_repeat_rate <= GATE_LONG_REPEAT_MAX,
             f"long_repeat={report.long_repeat_rate:.3f} > {GATE_LONG_REPEAT_MAX}")
    _need(native_stability_passed, "native stability gate failed")
    _need(gradient_path_ok, "gradient-path proof failed")
    _need(attribution_confirmed, "attribution test not confirmed")
    _need(not external_model_used, "external model participated")
    _need(not rewriting_system_used, "rewriting system participated")
    _need(not retrieval_answer_used, "retrieval-answer path participated")
    _need(not canned_response_used, "canned-response path participated")

    return PromotionDecision(
        passed=(len(reasons) == 0),
        reasons=reasons,
        gates={
            "E_stream": report.E_stream, "R_readable": report.R_readable,
            "R_one": report.R_one, "R_two": report.R_two,
            "continuity": report.continuity_rate,
            "R_fixation": report.R_fixation_on_unrelated,
            "long_repeat": report.long_repeat_rate,
            "architecture_delta_zero": architecture_delta_zero,
            "protected_p2_hash_unchanged": protected_p2_hash_unchanged,
            "native_stability_passed": native_stability_passed,
            "gradient_path_ok": gradient_path_ok,
            "attribution_confirmed": attribution_confirmed,
        })
