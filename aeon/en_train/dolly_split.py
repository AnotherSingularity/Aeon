"""aeon.en_train.dolly_split — deterministic grouped splitter for the
ENGLISH-PROOF-0 pilot.

Requirements from the directive Section 6:

  * Split BEFORE any optimizer step.
  * Deterministic grouped splitting: 90% train / 5% val / 5% sealed test.
  * Normalise and hash the combined instruction + context + response
    of each record.
  * Perform:
      - exact duplicate detection
      - record-ID collision detection
      - 5-gram Jaccard near-duplicate clustering, threshold 0.85
  * Near-duplicate records MUST remain in the same partition.
  * Write immutable manifests containing record hashes for all
    partitions. Write and hash the sealed-test manifest before
    training begins.

This module intentionally contains NO training code and NO network
access. It operates on a list of already-loaded (record_id, instruction,
context, response, category) tuples. The caller is responsible for
loading them from the immutable acquisition at
research-data/incoming/EN-DOLLY-15K/sources/... AFTER the operator has
uploaded the corpus and populated docs/en_train/dolly15k_provenance.json.
"""
from __future__ import annotations

import hashlib
import json
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Record + normalization
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DollyRecord:
    record_id: str
    instruction: str
    context: str            # may be empty
    response: str
    category: str

    def combined(self) -> str:
        parts = [self.instruction]
        if self.context:
            parts.append(self.context)
        parts.append(self.response)
        return "\n".join(parts)


def normalize_text(s: str) -> str:
    """Directive-permitted mechanical normalization only:
      * Unicode NFC
      * line-ending -> \\n
      * strip trailing whitespace from each line
      * collapse leading/trailing whitespace on the whole string
    No content is rewritten."""
    s = unicodedata.normalize("NFC", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    return s.strip()


def hash_normalized(record: DollyRecord) -> str:
    """Content hash used for exact-duplicate detection."""
    canon = normalize_text(record.combined())
    return "sha256:" + hashlib.sha256(canon.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Near-duplicate clustering — 5-gram Jaccard, threshold 0.85
# ---------------------------------------------------------------------------
NGRAM_N = 5
NEAR_DUP_THRESHOLD = 0.85


def _ngrams(text: str, n: int = NGRAM_N) -> Set[str]:
    tokens = normalize_text(text).lower().split()
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union


def cluster_near_duplicates(records: Sequence[DollyRecord],
                            threshold: float = NEAR_DUP_THRESHOLD
                            ) -> Dict[str, str]:
    """Return record_id -> cluster_id. Two records with 5-gram
    Jaccard >= threshold on their combined text share a cluster.
    Deterministic union-find; iteration ordered by record_id.

    Implementation uses an exact candidate-pair pruning strategy:
    two records with any non-zero Jaccard share at least one 5-gram,
    so we build a shingle -> record inverted index and only compare
    pairs that co-occur under at least one shingle. This is exact
    (never misses a >=threshold pair) and reduces the naive O(N^2)
    cost to O(N * avg_neighbours) for typical text corpora.
    """
    order = sorted(records, key=lambda r: r.record_id)
    grams = [_ngrams(r.combined()) for r in order]
    parent: Dict[str, str] = {r.record_id: r.record_id for r in order}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    # Shingle -> list of record indices. Deterministic since `order`
    # is sorted by record_id.
    index: Dict[str, List[int]] = {}
    for i, gs in enumerate(grams):
        for g in gs:
            index.setdefault(g, []).append(i)

    # For each record i, gather candidate partners j > i via any
    # shared shingle, then check exact Jaccard once per pair.
    for i in range(len(order)):
        candidates: Set[int] = set()
        for g in grams[i]:
            for j in index.get(g, ()):
                if j > i:
                    candidates.add(j)
        for j in candidates:
            if _jaccard(grams[i], grams[j]) >= threshold:
                union(order[i].record_id, order[j].record_id)

    return {rid: find(rid) for rid in parent}


# ---------------------------------------------------------------------------
# Deterministic grouped split
# ---------------------------------------------------------------------------
@dataclass
class SplitReport:
    total_records: int
    train_ids: List[str]
    val_ids: List[str]
    sealed_test_ids: List[str]
    exact_duplicate_groups: List[List[str]]        # each group = duplicate ids
    record_id_collisions: List[str]                 # rid appeared multiple times
    near_duplicate_clusters: Dict[str, List[str]]  # cluster_id -> members
    excluded_records: List[Dict[str, str]]          # {record_id, reason}
    split_seed: int
    ngram_n: int
    near_duplicate_threshold: float

    def to_dict(self) -> Dict:
        return {
            "total_records": self.total_records,
            "train_count": len(self.train_ids),
            "val_count": len(self.val_ids),
            "sealed_test_count": len(self.sealed_test_ids),
            "train_ids": self.train_ids,
            "val_ids": self.val_ids,
            "sealed_test_ids": self.sealed_test_ids,
            "exact_duplicate_groups": self.exact_duplicate_groups,
            "record_id_collisions": self.record_id_collisions,
            "near_duplicate_clusters": {
                cid: sorted(members)
                for cid, members in sorted(self.near_duplicate_clusters.items())
            },
            "excluded_records": sorted(self.excluded_records,
                                         key=lambda x: x["record_id"]),
            "split_seed": self.split_seed,
            "ngram_n": self.ngram_n,
            "near_duplicate_threshold": self.near_duplicate_threshold,
        }


def _stable_bucket(cluster_id: str, seed: int) -> float:
    """Map a cluster_id to a stable [0, 1) float using SHA-256(seed || cluster).
    Deterministic across machines and Python versions."""
    h = hashlib.sha256(f"{seed}|{cluster_id}".encode("utf-8")).digest()
    # Use the first 8 bytes as an unsigned int / 2**64
    v = int.from_bytes(h[:8], "big")
    return v / float(1 << 64)


def deterministic_split(records: Sequence[DollyRecord],
                        *,
                        train_frac: float = 0.90,
                        val_frac: float = 0.05,
                        sealed_frac: float = 0.05,
                        seed: int = 20260822,
                        ) -> SplitReport:
    """Deterministic grouped 90 / 5 / 5 split.

    * Exact duplicates (by content hash) form a group and are ALL kept
      together AND recorded in exact_duplicate_groups. The first
      lexicographic record_id in each group is the group representative;
      duplicates are recorded as excluded_records with reason
      "exact_duplicate_of_group_representative" so training sees each
      canonical text once.
    * Record-id collisions are recorded and the second and subsequent
      occurrences are excluded with reason "record_id_collision".
    * Near-duplicates (5-gram Jaccard >= 0.85) form clusters and stay
      in the same partition.
    * Assignment: each near-duplicate cluster is mapped to a stable
      [0,1) bucket using SHA-256(seed || cluster_id); ranges are
      [0, train_frac) train, [train_frac, train_frac+val_frac) val,
      remainder sealed_test.
    """
    assert abs(train_frac + val_frac + sealed_frac - 1.0) < 1e-9

    # 1. record-id collisions
    seen_ids: Set[str] = set()
    id_collisions: List[str] = []
    unique_records: List[DollyRecord] = []
    excluded: List[Dict[str, str]] = []
    for r in records:
        if r.record_id in seen_ids:
            id_collisions.append(r.record_id)
            excluded.append({"record_id": r.record_id,
                             "reason": "record_id_collision"})
            continue
        seen_ids.add(r.record_id)
        unique_records.append(r)

    # 2. exact duplicates by content hash
    by_hash: Dict[str, List[DollyRecord]] = {}
    for r in unique_records:
        by_hash.setdefault(hash_normalized(r), []).append(r)
    exact_dup_groups: List[List[str]] = []
    kept: List[DollyRecord] = []
    for h, group in by_hash.items():
        group_sorted = sorted(group, key=lambda x: x.record_id)
        if len(group_sorted) > 1:
            exact_dup_groups.append([r.record_id for r in group_sorted])
            rep = group_sorted[0]
            kept.append(rep)
            for r in group_sorted[1:]:
                excluded.append({
                    "record_id": r.record_id,
                    "reason": f"exact_duplicate_of_group_representative:{rep.record_id}",
                })
        else:
            kept.append(group_sorted[0])

    # 3. near-duplicate clusters over the kept set
    cluster_of = cluster_near_duplicates(kept)
    clusters: Dict[str, List[str]] = {}
    for rid, cid in cluster_of.items():
        clusters.setdefault(cid, []).append(rid)

    # 4. deterministic cluster-level partition assignment
    train_ids: List[str] = []
    val_ids: List[str] = []
    sealed_ids: List[str] = []
    for cid, members in clusters.items():
        b = _stable_bucket(cid, seed)
        if b < train_frac:
            train_ids.extend(members)
        elif b < train_frac + val_frac:
            val_ids.extend(members)
        else:
            sealed_ids.extend(members)

    return SplitReport(
        total_records=len(records),
        train_ids=sorted(train_ids),
        val_ids=sorted(val_ids),
        sealed_test_ids=sorted(sealed_ids),
        exact_duplicate_groups=[sorted(g) for g in exact_dup_groups],
        record_id_collisions=sorted(set(id_collisions)),
        near_duplicate_clusters=clusters,
        excluded_records=excluded,
        split_seed=seed,
        ngram_n=NGRAM_N,
        near_duplicate_threshold=NEAR_DUP_THRESHOLD,
    )


# ---------------------------------------------------------------------------
# Sealed-test manifest hash — sealed BEFORE training begins
# ---------------------------------------------------------------------------
def write_split_manifest(report: SplitReport, out_path: Path) -> Dict:
    """Write the split manifest AND sign the sealed-test partition
    with a SHA-256 hash so any later mutation is detectable. Returns
    the manifest dict actually written."""
    d = report.to_dict()
    d["generated_at_utc_epoch"] = time.time()
    # Sealed-test lock: hash the sorted sealed-test id list itself.
    sealed_canon = "\n".join(sorted(report.sealed_test_ids)).encode("utf-8")
    d["sealed_test_lock_sha256"] = "sha256:" + hashlib.sha256(sealed_canon).hexdigest()
    d["sealed_test_lock_scope"] = "sha256_of_sorted_sealed_test_ids"
    d["sealed_test_lock_rule"] = (
        "sealed_test_ids is sealed at split time; the pilot must "
        "abort if this hash no longer matches the sorted list of "
        "sealed_test_ids before evaluating the sealed partition.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(d, sort_keys=True, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    return d


def verify_sealed_test_lock(manifest_path: Path) -> Tuple[bool, str]:
    """Verify the sealed-test lock is intact. Returns (ok, message)."""
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    sealed = sorted(m.get("sealed_test_ids", []))
    canon = "\n".join(sealed).encode("utf-8")
    got = "sha256:" + hashlib.sha256(canon).hexdigest()
    want = m.get("sealed_test_lock_sha256")
    if want is None:
        return False, "sealed_test_lock_sha256 missing"
    if want != got:
        return False, f"sealed_test_lock_sha256 mismatch: want={want} got={got}"
    return True, "ok"
