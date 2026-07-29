"""
aeon/audit.py — hash-chained audit records.

Directive F3.4 wants audit continuity: a break in the chain is detected.
Implementation: each event carries `seq` (monotone), `prev_hash` (of the
previous event or "" for genesis), and its own `hash`. Reading the log verifies
the chain from head to tail.

Torch-free; safe to import at any layer.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, Iterator, Optional


def _sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


class AuditWriter:
    """Append-only writer for the hash-chained audit log.

    On the FIRST write, reads the existing log (if any) to recover the last seq
    and last hash, so restart continuity is preserved. Failed writes raise —
    audit is safety-critical per §16.2 (fail closed on audit-storage failure).
    Callers may catch and route to F5 SAFE_HALT.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._seq, self._last = self._recover()

    def _recover(self):
        if not os.path.exists(self.path):
            return 0, ""
        seq, last = 0, ""
        with open(self.path) as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    seq = int(rec["seq"])
                    last = rec["hash"]
        return seq, last

    def write(self, kind: str, **payload) -> Dict[str, Any]:
        seq = self._seq + 1
        rec = {"seq": seq, "kind": kind, "prev_hash": self._last, "ts": time.time(),
               "payload": payload}
        rec["hash"] = _sha256_hex(json.dumps(rec, sort_keys=True,
                                              separators=(",", ":")).encode("utf-8"))
        with open(self.path, "a") as fh:
            fh.write(json.dumps(rec, separators=(",", ":")) + "\n")
        self._seq, self._last = seq, rec["hash"]
        return rec


def read_audit(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def verify_chain(path: str) -> Optional[str]:
    """Verify prev_hash + monotone seq + individual hashes. Returns the first
    error message on failure, or None on success (empty log is valid)."""
    prev_hash = ""
    prev_seq = 0
    for rec in read_audit(path):
        expected_prev = prev_hash
        if rec.get("prev_hash", "") != expected_prev:
            return f"seq {rec.get('seq')}: prev_hash mismatch (expected {expected_prev!r}, got {rec.get('prev_hash')!r})"
        if int(rec.get("seq", 0)) != prev_seq + 1:
            return f"seq gap at seq={rec.get('seq')} (expected {prev_seq + 1})"
        base = {k: v for k, v in rec.items() if k != "hash"}
        expected_hash = _sha256_hex(json.dumps(base, sort_keys=True,
                                                separators=(",", ":")).encode("utf-8"))
        if expected_hash != rec.get("hash"):
            return f"seq {rec.get('seq')}: content hash mismatch"
        prev_seq = int(rec["seq"])
        prev_hash = rec["hash"]
    return None
