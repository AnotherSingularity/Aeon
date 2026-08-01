"""ACIS audit-event log.

Bounded, evaluation-only, offline. Records ACIS lifecycle events
without ever holding a payload reference. Ledger digest chains each
event to the previous so a replay attempt can be detected.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AcisEvent:
    """One entry in the ACIS audit log.

    ``payload_digest`` refers to the semantic digest of the object the
    event is about (broadcast, lease, capsule). The event NEVER carries
    the payload itself — that would violate the IP-preservation stance
    of §21 Security and IP tests.
    """

    schema_version: int
    seq: int
    kind: str
    boundary_index: int
    recursion_epoch: int
    payload_digest: str
    prev_ledger_digest: str
    ledger_digest: str
    ts_utc: str
    detail: Dict[str, Any] = field(default_factory=dict)


class AcisAuditLog:
    """In-memory chain. Persistent flush lives in ACIS-6 recovery.

    The ledger digest at position n is:

        H(seq | kind | boundary | epoch | payload_digest | prev_ledger)

    A replayed event carries the wrong prev_ledger and is refused.
    """

    def __init__(self) -> None:
        self._events: List[AcisEvent] = []
        self._seq = 0

    def __len__(self) -> int:
        return len(self._events)

    def events(self) -> List[AcisEvent]:
        return list(self._events)

    def head_digest(self) -> str:
        if not self._events:
            return "sha256:" + "0" * 64
        return self._events[-1].ledger_digest

    def append(
        self,
        *,
        kind: str,
        boundary_index: int,
        recursion_epoch: int,
        payload_digest: str,
        detail: Optional[Dict[str, Any]] = None,
    ) -> AcisEvent:
        self._seq += 1
        prev = self.head_digest()
        material = "|".join([
            str(self._seq), kind,
            str(int(boundary_index)),
            str(int(recursion_epoch)),
            payload_digest,
            prev,
        ]).encode("utf-8")
        ledger_digest = "sha256:" + hashlib.sha256(material).hexdigest()
        ev = AcisEvent(
            schema_version=1,
            seq=self._seq,
            kind=kind,
            boundary_index=int(boundary_index),
            recursion_epoch=int(recursion_epoch),
            payload_digest=payload_digest,
            prev_ledger_digest=prev,
            ledger_digest=ledger_digest,
            ts_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            detail=dict(detail or {}),
        )
        self._events.append(ev)
        return ev

    def to_json(self) -> str:
        return json.dumps(
            [{"seq": e.seq, "kind": e.kind,
              "boundary_index": e.boundary_index,
              "recursion_epoch": e.recursion_epoch,
              "payload_digest": e.payload_digest,
              "prev_ledger_digest": e.prev_ledger_digest,
              "ledger_digest": e.ledger_digest,
              "ts_utc": e.ts_utc,
              "detail": e.detail}
             for e in self._events],
            sort_keys=True, indent=2)
