"""aeon.desktop.protocol — versioned event + request + error schemas.

Every event / request is a frozen dataclass. Nothing here carries raw
tensors, activations, filesystem paths beyond what the caller passed,
or model internals. Payloads are safe to render.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


PROTOCOL_VERSION = 1


# ---------------------------------------------------------------------------
# Event kinds (per §18)
# ---------------------------------------------------------------------------
class EventKind(str, Enum):
    RUNTIME_STARTING = "runtime_starting"
    RUNTIME_PREFLIGHT_STARTED = "runtime_preflight_started"
    RUNTIME_PREFLIGHT_COMPLETED = "runtime_preflight_completed"
    MODEL_VALIDATION_STARTED = "model_validation_started"
    MODEL_VALIDATION_COMPLETED = "model_validation_completed"
    MODEL_LOADING_STARTED = "model_loading_started"
    MODEL_LOADING_PROGRESS = "model_loading_progress"
    MODEL_READY = "model_ready"
    SESSION_CREATED = "session_created"
    SESSION_RESET = "session_reset"
    SESSION_CLOSED = "session_closed"
    GENERATION_QUEUED = "generation_queued"
    GENERATION_STARTED = "generation_started"
    TOKEN_GENERATED = "token_generated"
    TEXT_DELTA = "text_delta"
    GENERATION_PROGRESS = "generation_progress"
    GENERATION_COMPLETED = "generation_completed"
    GENERATION_CANCEL_REQUESTED = "generation_cancel_requested"
    GENERATION_CANCELLED = "generation_cancelled"
    GENERATION_FAILED = "generation_failed"
    RUNTIME_WARNING = "runtime_warning"
    RUNTIME_FAILED = "runtime_failed"
    RUNTIME_SHUTDOWN_STARTED = "runtime_shutdown_started"
    RUNTIME_SHUTDOWN_COMPLETED = "runtime_shutdown_completed"


# ---------------------------------------------------------------------------
# Structured error codes (per §28)
# ---------------------------------------------------------------------------
class ErrorCode(str, Enum):
    RELEASE_MANIFEST_MISSING = "RELEASE_MANIFEST_MISSING"
    RELEASE_MANIFEST_INVALID = "RELEASE_MANIFEST_INVALID"
    MODEL_MISSING = "MODEL_MISSING"
    MODEL_DIGEST_MISMATCH = "MODEL_DIGEST_MISMATCH"
    MODEL_SCHEMA_UNSUPPORTED = "MODEL_SCHEMA_UNSUPPORTED"
    MODEL_CONFIG_MISMATCH = "MODEL_CONFIG_MISMATCH"
    TOKENIZER_MISSING = "TOKENIZER_MISSING"
    TOKENIZER_DIGEST_MISMATCH = "TOKENIZER_DIGEST_MISMATCH"
    TOKENIZER_VOCAB_MISMATCH = "TOKENIZER_VOCAB_MISMATCH"
    ARCHITECTURE_MISMATCH = "ARCHITECTURE_MISMATCH"
    FIXED_K_MISMATCH = "FIXED_K_MISMATCH"
    RUNTIME_START_FAILED = "RUNTIME_START_FAILED"
    MODEL_LOAD_FAILED = "MODEL_LOAD_FAILED"
    PROMPT_EMPTY = "PROMPT_EMPTY"
    PROMPT_TOO_LARGE = "PROMPT_TOO_LARGE"
    SETTINGS_INVALID = "SETTINGS_INVALID"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    REQUEST_ALREADY_ACTIVE = "REQUEST_ALREADY_ACTIVE"
    GENERATION_FAILED = "GENERATION_FAILED"
    GENERATION_CANCELLED = "GENERATION_CANCELLED"
    RUNTIME_UNRESPONSIVE = "RUNTIME_UNRESPONSIVE"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    SHUTDOWN_TIMEOUT = "SHUTDOWN_TIMEOUT"


# ---------------------------------------------------------------------------
# Runtime state machine (per §21)
# ---------------------------------------------------------------------------
class RuntimeState(str, Enum):
    NOT_STARTED = "NOT_STARTED"
    STARTING = "STARTING"
    PREFLIGHT = "PREFLIGHT"
    VALIDATING_RELEASE = "VALIDATING_RELEASE"
    LOADING_MODEL = "LOADING_MODEL"
    READY = "READY"
    GENERATING = "GENERATING"
    CANCELLING = "CANCELLING"
    FAILED = "FAILED"
    SHUTTING_DOWN = "SHUTTING_DOWN"
    STOPPED = "STOPPED"


# Allowed transitions. Rejecting the impossible ones is the state machine's
# job — see aeon.desktop.runtime.
_ALLOWED = {
    RuntimeState.NOT_STARTED: {RuntimeState.STARTING},
    RuntimeState.STARTING: {RuntimeState.PREFLIGHT, RuntimeState.FAILED},
    RuntimeState.PREFLIGHT: {RuntimeState.VALIDATING_RELEASE, RuntimeState.FAILED},
    RuntimeState.VALIDATING_RELEASE: {RuntimeState.LOADING_MODEL, RuntimeState.FAILED},
    RuntimeState.LOADING_MODEL: {RuntimeState.READY, RuntimeState.FAILED},
    RuntimeState.READY: {RuntimeState.GENERATING, RuntimeState.SHUTTING_DOWN,
                          RuntimeState.FAILED},
    RuntimeState.GENERATING: {RuntimeState.READY, RuntimeState.CANCELLING,
                                RuntimeState.FAILED, RuntimeState.SHUTTING_DOWN},
    RuntimeState.CANCELLING: {RuntimeState.READY, RuntimeState.FAILED,
                                RuntimeState.SHUTTING_DOWN},
    RuntimeState.FAILED: {RuntimeState.SHUTTING_DOWN, RuntimeState.STARTING},
    RuntimeState.SHUTTING_DOWN: {RuntimeState.STOPPED},
    RuntimeState.STOPPED: set(),
}


def is_valid_transition(cur: RuntimeState, nxt: RuntimeState) -> bool:
    return nxt in _ALLOWED[cur]


# ---------------------------------------------------------------------------
# Generation settings + bounds (per §14)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationBounds:
    max_new_tokens_min: int = 1
    max_new_tokens_max: int = 1024
    max_new_tokens_default: int = 128
    temperature_min: float = 0.0
    temperature_max: float = 2.0
    temperature_default: float = 0.8
    top_p_min: float = 0.05
    top_p_max: float = 1.0
    top_p_default: float = 0.95
    top_k_min: int = 0
    top_k_max: int = 500
    top_k_default: int = 50
    repetition_penalty_min: float = 1.0
    repetition_penalty_max: float = 2.0
    repetition_penalty_default: float = 1.1


BOUNDS = GenerationBounds()


@dataclass(frozen=True)
class GenerationOptions:
    max_new_tokens: int = BOUNDS.max_new_tokens_default
    temperature: float = BOUNDS.temperature_default
    top_p: float = BOUNDS.top_p_default
    top_k: int = BOUNDS.top_k_default
    repetition_penalty: float = BOUNDS.repetition_penalty_default
    deterministic_seed: Optional[int] = None


class SettingsInvalid(RuntimeError):
    def __init__(self, code: ErrorCode, detail: str):
        super().__init__(f"{code.value}: {detail}")
        self.code = code
        self.detail = detail


def validate_settings(opts: GenerationOptions,
                          vocab_size: int) -> GenerationOptions:
    """Enforce the versioned bounds. Rejects NaN, infinity, wrong types,
    and out-of-range values."""
    import math
    def _bad(name, v):
        return not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v)

    if _bad("max_new_tokens", opts.max_new_tokens) or int(opts.max_new_tokens) != opts.max_new_tokens:
        raise SettingsInvalid(ErrorCode.SETTINGS_INVALID,
                                  f"max_new_tokens must be an integer, got {opts.max_new_tokens!r}")
    if not (BOUNDS.max_new_tokens_min <= opts.max_new_tokens <= BOUNDS.max_new_tokens_max):
        raise SettingsInvalid(ErrorCode.SETTINGS_INVALID,
                                  f"max_new_tokens out of range [{BOUNDS.max_new_tokens_min}, {BOUNDS.max_new_tokens_max}]")
    if _bad("temperature", opts.temperature):
        raise SettingsInvalid(ErrorCode.SETTINGS_INVALID, "temperature invalid")
    if not (BOUNDS.temperature_min <= opts.temperature <= BOUNDS.temperature_max):
        raise SettingsInvalid(ErrorCode.SETTINGS_INVALID,
                                  f"temperature out of range [{BOUNDS.temperature_min}, {BOUNDS.temperature_max}]")
    if _bad("top_p", opts.top_p):
        raise SettingsInvalid(ErrorCode.SETTINGS_INVALID, "top_p invalid")
    if not (BOUNDS.top_p_min <= opts.top_p <= BOUNDS.top_p_max):
        raise SettingsInvalid(ErrorCode.SETTINGS_INVALID, "top_p out of range")
    if _bad("top_k", opts.top_k) or int(opts.top_k) != opts.top_k:
        raise SettingsInvalid(ErrorCode.SETTINGS_INVALID, "top_k must be int")
    top_k_upper = min(BOUNDS.top_k_max, vocab_size)
    if not (BOUNDS.top_k_min <= opts.top_k <= top_k_upper):
        raise SettingsInvalid(ErrorCode.SETTINGS_INVALID, "top_k out of range")
    if _bad("repetition_penalty", opts.repetition_penalty):
        raise SettingsInvalid(ErrorCode.SETTINGS_INVALID, "repetition_penalty invalid")
    if not (BOUNDS.repetition_penalty_min <= opts.repetition_penalty
              <= BOUNDS.repetition_penalty_max):
        raise SettingsInvalid(ErrorCode.SETTINGS_INVALID, "repetition_penalty out of range")
    return opts


# ---------------------------------------------------------------------------
# Runtime event
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RuntimeEvent:
    event_id: str
    event_type: EventKind
    created_at: float
    sequence_number: int
    schema_version: int = PROTOCOL_VERSION
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "created_at": self.created_at,
            "sequence_number": self.sequence_number,
            "session_id": self.session_id,
            "request_id": self.request_id,
            "payload": dict(self.payload),
        }


def make_event(kind: EventKind, *, seq: int, session_id: Optional[str] = None,
                 request_id: Optional[str] = None, payload: Optional[Dict[str, Any]] = None
                 ) -> RuntimeEvent:
    return RuntimeEvent(
        event_id=str(uuid.uuid4()),
        event_type=kind,
        created_at=time.time(),
        sequence_number=int(seq),
        session_id=session_id,
        request_id=request_id,
        payload=dict(payload or {}),
    )
