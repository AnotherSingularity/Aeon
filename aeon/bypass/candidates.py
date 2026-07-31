"""L2 same-visible-state candidate-pair search.

Identifies positions in a corpus where two occurrences share a visible
prefix (or a declared visible projection) closely enough that any
downstream causal test can compare their Recursion trajectories on a
matched-visible-state basis. The search MUST NOT read the hidden
Recursion state — otherwise the candidate set is selected to favour
a specific hidden-state result, which is a form of coordinate p-hacking
the L-series' claim ladder explicitly forbids.

The output is a frozen ``CandidateSet`` with:

    * A stable ``candidate_set_digest`` (SHA-256 of the sorted pair
      list) so downstream L3+ tests bind to a specific set.
    * A ``locked_at`` timestamp — set at construction; the file
      writer refuses to overwrite an existing locked set.

L2's tests exercise the search on the bounded synthetic-English
fixture. No scientific claim is drawn from the L2 candidate set;
L3+ callibration uses candidate sets built from the real vendored
corpus.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, List, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class CandidatePair:
    """One (i, j) pair: two positions with matching visible state."""

    left_record_id: str
    left_position: int
    right_record_id: str
    right_position: int
    visible_distance: float
    match_method: str  # "exact_prefix" | "declared_projection"


@dataclass(frozen=True)
class CandidateSet:
    schema_version: int
    match_method: str
    epsilon: float
    prefix_length: Optional[int]
    projection_id: Optional[str]
    pairs: Tuple[CandidatePair, ...]
    candidate_set_digest: str
    locked_at: str  # ISO date/time
    notes: str = ""


class CandidateSearchError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# Exact-prefix search
# ---------------------------------------------------------------------------
def find_exact_prefix_matches(
    records: Sequence[Mapping[str, Any]],
    *,
    prefix_length: int,
    max_pairs: int = 4096,
) -> List[CandidatePair]:
    """Return all pairs of positions (i, j) where the ``prefix_length``
    tokens immediately preceding position i match exactly the
    ``prefix_length`` tokens preceding position j.

    ``records`` is a sequence of dicts with at least
    ``{"record_id": str, "tokens": Sequence[int]}``.

    Only VISIBLE information (the token stream) is used. No hidden
    state is read.
    """
    if prefix_length < 1:
        raise CandidateSearchError(
            "invalid_prefix_length", f"prefix_length={prefix_length}")
    # (prefix_tuple, record_id, position) — one row per candidate site
    seen: dict = {}
    pairs: List[CandidatePair] = []
    for rec in records:
        rid = rec["record_id"]
        toks = rec["tokens"]
        n = len(toks)
        for pos in range(prefix_length, n):
            prefix = tuple(toks[pos - prefix_length: pos])
            if prefix in seen:
                for (other_rid, other_pos) in seen[prefix]:
                    pairs.append(CandidatePair(
                        left_record_id=other_rid,
                        left_position=other_pos,
                        right_record_id=rid,
                        right_position=pos,
                        visible_distance=0.0,
                        match_method="exact_prefix",
                    ))
                    if len(pairs) >= max_pairs:
                        return pairs
                seen[prefix].append((rid, pos))
            else:
                seen[prefix] = [(rid, pos)]
    return pairs


# ---------------------------------------------------------------------------
# Declared-projection search
# ---------------------------------------------------------------------------
def find_projection_matches(
    projections: Sequence[Mapping[str, Any]],
    *,
    epsilon: float,
    projection_id: str,
    max_pairs: int = 4096,
) -> List[CandidatePair]:
    """Pair sites whose declared VISIBLE projection vectors are within
    epsilon of each other. ``projections`` is a sequence of dicts with
    ``{"record_id": str, "position": int, "vector": Sequence[float]}``
    — the caller pre-computes the projection from visible signals
    only (e.g. a learned projection over pre-broadcast token features).

    O(n²) exhaustive comparison. Adequate for L2's bounded fixture;
    L3+ swaps in an ANN index when the corpus size warrants it.
    """
    n = len(projections)
    pairs: List[CandidatePair] = []
    for i in range(n):
        vi = projections[i]["vector"]
        ri = projections[i]["record_id"]
        pi = projections[i]["position"]
        for j in range(i + 1, n):
            vj = projections[j]["vector"]
            d = _l2(vi, vj)
            if d <= epsilon:
                pairs.append(CandidatePair(
                    left_record_id=ri,
                    left_position=pi,
                    right_record_id=projections[j]["record_id"],
                    right_position=projections[j]["position"],
                    visible_distance=float(d),
                    match_method="declared_projection",
                ))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _l2(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise CandidateSearchError(
            "shape_mismatch", f"{len(a)} vs {len(b)}")
    s = 0.0
    for u, v in zip(a, b):
        d = float(u) - float(v)
        s += d * d
    return s ** 0.5


# ---------------------------------------------------------------------------
# Locking and evidence
# ---------------------------------------------------------------------------
def build_locked_set(
    pairs: Sequence[CandidatePair],
    *,
    match_method: str,
    epsilon: float,
    prefix_length: Optional[int],
    projection_id: Optional[str],
    locked_at_iso: str,
    notes: str = "",
) -> CandidateSet:
    """Wrap ``pairs`` in an immutable CandidateSet with a
    content-derived digest so downstream L3+ code can bind to a
    specific set."""
    payload = json.dumps(
        [(p.left_record_id, p.left_position,
          p.right_record_id, p.right_position,
          p.visible_distance, p.match_method)
         for p in pairs],
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    return CandidateSet(
        schema_version=1,
        match_method=match_method,
        epsilon=float(epsilon),
        prefix_length=prefix_length,
        projection_id=projection_id,
        pairs=tuple(sorted(pairs, key=lambda p: (
            p.left_record_id, p.left_position,
            p.right_record_id, p.right_position))),
        candidate_set_digest=digest,
        locked_at=locked_at_iso,
        notes=notes,
    )
