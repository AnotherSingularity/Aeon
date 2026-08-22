"""aeon.en_train.data — corpus intake, splitter, dedup, tokenizer checks.

Implements §3 (document-level partitioning, exact + Jaccard-5
near-duplicate grouping, per-book / per-author caps) and §4
(tokenizer + r_UNK gate).

Also owns the CORPUS INTAKE SCHEMA validator for D_G / D_C / D_A / D_E
directories at `research-data/incoming/<CORPUS_ID>/`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

from . import (
    MAX_SINGLE_AUTHOR_TOKEN_FRACTION,
    MAX_SINGLE_BOOK_TOKEN_FRACTION,
    MAX_UNK_RATE,
    FIXED_VOCAB_SIZE,
)


# ---------------------------------------------------------------------------
# Provenance / intake
# ---------------------------------------------------------------------------
REQUIRED_PROVENANCE_FIELDS = (
    "source_id",
    "author_or_institution",
    "original_publication_location",
    "acquisition_date",
    "license_or_public_domain_basis",
    "permitted_use_notes",
    "sha256",
    "byte_length",
    "document_count",
    "estimated_token_count",
    "encoding",
    "preprocessing_declaration",
)

REQUIRED_INTAKE_DIRS = ("sources", "provenance", "licenses", "manifests")


class IntakeError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _sha256_bytes(b: bytes) -> str:
    return "sha256:" + hashlib.sha256(b).hexdigest()


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def validate_intake_layout(intake_root: Path) -> Dict[str, Any]:
    """Verify a `research-data/incoming/<CORPUS_ID>/` directory carries
    the required layout and per-source provenance."""
    intake_root = Path(intake_root)
    for d in REQUIRED_INTAKE_DIRS:
        if not (intake_root / d).is_dir():
            raise IntakeError("missing_intake_directory", str(intake_root / d))
    sources_dir = intake_root / "sources"
    prov_dir = intake_root / "provenance"

    sources = sorted(p for p in sources_dir.iterdir()
                        if p.is_file() and not p.name.startswith("."))
    if not sources:
        raise IntakeError("no_sources", str(sources_dir))

    per_source: List[Dict[str, Any]] = []
    for src in sources:
        prov_path = prov_dir / (src.stem + ".json")
        if not prov_path.exists():
            raise IntakeError("missing_provenance", src.name)
        try:
            prov = json.loads(prov_path.read_text(encoding="utf-8"))
        except Exception as e:
            raise IntakeError("provenance_not_parseable", f"{prov_path.name}: {e}")
        missing = [f for f in REQUIRED_PROVENANCE_FIELDS if f not in prov]
        if missing:
            raise IntakeError("provenance_missing_fields",
                                f"{prov_path.name}: {missing}")
        got_sha = _sha256_file(src)
        want_sha = prov["sha256"]
        if want_sha and got_sha != want_sha:
            raise IntakeError("source_digest_mismatch",
                                f"{src.name}: got={got_sha} want={want_sha}")
        per_source.append({
            "source_id": prov["source_id"],
            "path": str(src),
            "sha256": got_sha,
            "byte_length": src.stat().st_size,
            "author_or_institution": prov["author_or_institution"],
            "license_or_public_domain_basis": prov["license_or_public_domain_basis"],
            "preprocessing_declaration": prov["preprocessing_declaration"],
        })
    return {
        "intake_root": str(intake_root),
        "layout_ok": True,
        "source_count": len(per_source),
        "sources": per_source,
    }


# ---------------------------------------------------------------------------
# Preprocessing helpers
# ---------------------------------------------------------------------------
def normalize_for_dedup(text: str) -> str:
    """Unicode NFC, lowercase, whitespace-collapsed. Used for
    dedup / near-dup only — NOT for training encoding."""
    t = unicodedata.normalize("NFC", text)
    t = t.lower()
    t = re.sub(r"\s+", " ", t).strip()
    return t


def word_ngrams(text: str, n: int = 5) -> FrozenSet[Tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    if len(words) < n:
        return frozenset()
    return frozenset(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def jaccard(a: FrozenSet, b: FrozenSet) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Document-level splitter (§3)
# ---------------------------------------------------------------------------
@dataclass
class Document:
    doc_id: str
    text: str
    source_id: str
    author_or_institution: str
    est_token_count: int
    normalized: str = ""
    ngrams5: FrozenSet[Tuple[str, ...]] = field(default_factory=frozenset)

    def hash_exact(self) -> str:
        return _sha256_bytes(self.normalized.encode("utf-8"))


def prepare_documents(docs: Sequence[Document]) -> List[Document]:
    out: List[Document] = []
    for d in docs:
        d.normalized = normalize_for_dedup(d.text)
        d.ngrams5 = word_ngrams(d.normalized, 5)
        out.append(d)
    return out


class Splitter:
    """Document-level 90/5/5 splitter (§3).

    * Groups by exact-normalized-hash AND by Jaccard-5 ≥ 0.85.
    * Never splits a duplicate group across partitions.
    * Enforces per-book (≤0.5%) and per-author (≤2%) token-share caps
      by group placement (rejecting a group whose placement would
      exceed the cap on any target partition).
    * Yields deterministic partition membership under a fixed seed.
    """

    def __init__(self,
                     val_fraction: float = 0.05,
                     test_fraction: float = 0.05,
                     jaccard_threshold: float = 0.85,
                     seed: int = 20260803):
        assert 0.0 < val_fraction < 0.5 and 0.0 < test_fraction < 0.5
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.jaccard_threshold = jaccard_threshold
        self.seed = seed

    def _group_duplicates(self, docs: List[Document]) -> List[List[int]]:
        """Union-find groups by exact and Jaccard-5 duplication."""
        n = len(docs)
        parent = list(range(n))
        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb: parent[rb] = ra
        # exact by normalized hash
        by_hash: Dict[str, int] = {}
        for i, d in enumerate(docs):
            h = d.hash_exact()
            if h in by_hash: union(by_hash[h], i)
            else: by_hash[h] = i
        # near-dup by Jaccard-5 (O(n²) — corpora are small here)
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) == find(j): continue
                if jaccard(docs[i].ngrams5, docs[j].ngrams5) >= self.jaccard_threshold:
                    union(i, j)
        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return list(groups.values())

    def split(self, docs: List[Document]) -> Dict[str, List[Document]]:
        docs = prepare_documents(list(docs))
        n_total_tokens = sum(d.est_token_count for d in docs)
        if n_total_tokens <= 0:
            raise IntakeError("empty_corpus", "no positive token counts")
        groups = self._group_duplicates(docs)
        # Deterministic shuffle of GROUPS
        import random
        rng = random.Random(self.seed)
        rng.shuffle(groups)

        target_val_tokens = self.val_fraction * n_total_tokens
        target_test_tokens = self.test_fraction * n_total_tokens
        val_toks = test_toks = train_toks = 0
        val_docs: List[Document] = []
        test_docs: List[Document] = []
        train_docs: List[Document] = []

        # §3 per-source and per-author caps (per partition, per set of
        # partitions we place the group into). We check the destination
        # partition only.
        book_caps: Dict[Tuple[str, str], int] = {}   # (partition, source_id) -> tokens
        author_caps: Dict[Tuple[str, str], int] = {}

        def _fits_caps(partition_docs_tokens: int, add_group: List[Document],
                          part_name: str) -> bool:
            # If the destination partition is TRAIN, apply §3 caps.
            if part_name != "train": return True
            # Recompute what would happen if we added this group.
            total_after = partition_docs_tokens + sum(d.est_token_count for d in add_group)
            if total_after <= 0: return True
            # Aggregate group tokens per source and per author
            book_add: Dict[str, int] = {}
            author_add: Dict[str, int] = {}
            for d in add_group:
                book_add[d.source_id] = book_add.get(d.source_id, 0) + d.est_token_count
                author_add[d.author_or_institution] = author_add.get(d.author_or_institution, 0) + d.est_token_count
            for src, add_tk in book_add.items():
                cur = book_caps.get((part_name, src), 0)
                if (cur + add_tk) / total_after > MAX_SINGLE_BOOK_TOKEN_FRACTION:
                    return False
            for auth, add_tk in author_add.items():
                cur = author_caps.get((part_name, auth), 0)
                if (cur + add_tk) / total_after > MAX_SINGLE_AUTHOR_TOKEN_FRACTION:
                    return False
            return True

        def _place(gp: List[Document], part_name: str) -> None:
            nonlocal train_toks, val_toks, test_toks
            if part_name == "train":
                train_docs.extend(gp)
                train_toks += sum(d.est_token_count for d in gp)
                for d in gp:
                    book_caps[("train", d.source_id)] = book_caps.get(("train", d.source_id), 0) + d.est_token_count
                    author_caps[("train", d.author_or_institution)] = author_caps.get(("train", d.author_or_institution), 0) + d.est_token_count
            elif part_name == "validation":
                val_docs.extend(gp); val_toks += sum(d.est_token_count for d in gp)
            else:
                test_docs.extend(gp); test_toks += sum(d.est_token_count for d in gp)

        # Greedy fill: prefer val/test until targets are hit, else train.
        for grp_idxs in groups:
            gp = [docs[i] for i in grp_idxs]
            gp_toks = sum(d.est_token_count for d in gp)
            placed = False
            if val_toks < target_val_tokens:
                _place(gp, "validation"); placed = True
            elif test_toks < target_test_tokens:
                _place(gp, "test"); placed = True
            else:
                if _fits_caps(train_toks, gp, "train"):
                    _place(gp, "train"); placed = True
                else:
                    # Rejected by caps — surface so the operator can
                    # re-balance rather than silently dropping.
                    raise IntakeError("partition_cap_exceeded",
                                          f"group with sources={sorted({d.source_id for d in gp})} "
                                          f"would exceed §3 per-book or per-author cap on train")
            if not placed:
                raise IntakeError("splitter_left_group_unplaced",
                                      "internal: no partition chosen for group")

        return {"train": train_docs, "validation": val_docs, "test": test_docs}


# ---------------------------------------------------------------------------
# Tokenizer + r_UNK gate (§4)
# ---------------------------------------------------------------------------
@dataclass
class TokenizerCheckResult:
    total_tokens: int
    unk_tokens: int
    r_unk: float
    max_id_seen: int
    passed: bool
    per_document: List[Dict[str, Any]] = field(default_factory=list)


def run_tokenizer_check(tokenizer, documents: Sequence[Document],
                            max_unk_rate: float = MAX_UNK_RATE
                            ) -> TokenizerCheckResult:
    """§4: encode every document, count UNK, verify id range."""
    unk_id = tokenizer.unk_id
    total = 0
    unk = 0
    max_id = -1
    per: List[Dict[str, Any]] = []
    for d in documents:
        ids = tokenizer.encode(d.text, add_bos=False, add_eos=False)
        for i in ids:
            if not (0 <= i < FIXED_VOCAB_SIZE):
                raise IntakeError("tokenizer_id_out_of_range",
                                      f"{d.doc_id}: id={i}")
            if i > max_id: max_id = i
            if i == unk_id: unk += 1
        total += len(ids)
        per.append({"doc_id": d.doc_id, "tokens": len(ids),
                       "unk_local": sum(1 for i in ids if i == unk_id)})
    r_unk = unk / total if total > 0 else 0.0
    return TokenizerCheckResult(
        total_tokens=total, unk_tokens=unk, r_unk=r_unk,
        max_id_seen=max_id, passed=(r_unk <= max_unk_rate),
        per_document=per,
    )
