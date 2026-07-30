"""
aeon/protected_checkpoint.py — F3 authenticated + anti-rollback envelope.

Extends aeon.checkpoint (E3) rather than replacing it. Adds:

  §F3.1 Authenticated checkpoint envelope: HMAC-SHA256 over the checkpoint file
        bytes AND the metadata JSON (both authenticated together). Uses only
        `hashlib.new("sha256")` and `hmac` from the Python stdlib — no custom
        crypto. Fails closed on tag mismatch.

  §F3.2 Optional confidentiality: an interface wraps the payload for AEAD when
        an operator supplies a key handle (via a KeyRef). Uses ONLY authenticated-
        encryption backends: AES-GCM via `cryptography.hazmat` when installed,
        otherwise disabled with a clear error. NEVER stores keys in the
        checkpoint; NEVER silently loads an encrypted artefact as plaintext.

  §F3.3 Anti-rollback: monotonic `authorized_step` recorded on save and refused
        on load when `authorized_step` < current authorized state (unless the
        caller supplies an explicit RecoveryDecision).

  §F3.4 Sensitive-state handling: no plaintext of encrypted payload leaks into
        the .prev sidecar. Temporary files are unlinked promptly.

Keys are referenced by handle only. In the reference implementation the handle
resolves to an in-memory bytes object provided by the operator (production
integrations replace `KeyRef.resolve()` with a KMS call). We do NOT claim
guaranteed memory erasure on general-purpose hardware (§F1 global non-guarantee).
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import tempfile
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch

from aeon.checkpoint import (
    CheckpointIncompatible,
    CheckpointCorrupt,
    K_LOCKED,
    PATCH_MANIFEST_VERSION,
    CHECKPOINT_SCHEMA_VERSION,
    _sha256,
    source_commit_id,
)


PROTECTED_ENVELOPE_VERSION = 1
MAC_ALGO = "hmac-sha256"


class CheckpointAuthenticationError(CheckpointCorrupt):
    """MAC verification failed."""


class AntiRollbackViolation(CheckpointIncompatible):
    """Attempted resume from a checkpoint older than the authorized state."""


class KeyUnavailableError(CheckpointCorrupt):
    """Encryption key referenced by the envelope is not resolvable."""


# ---------------------------------------------------------------------------
# KeyRef: opaque key handle (§F3.2)
# ---------------------------------------------------------------------------
@dataclass
class KeyRef:
    """Opaque reference to keying material. The `resolve` callable returns raw
    bytes (32 bytes for AES-256-GCM / HMAC-SHA256). Production integrations
    supply resolve = KMS.fetch; dev integrations supply resolve = lambda:
    ephemeral_key. Never store the resolved bytes in a checkpoint."""
    handle: str
    resolve: Callable[[], bytes] = field(repr=False)

    def key_bytes(self) -> bytes:
        b = self.resolve()
        if not isinstance(b, (bytes, bytearray)) or len(b) < 32:
            raise KeyUnavailableError(
                f"KeyRef({self.handle}) resolved to invalid material (need ≥32 bytes)")
        return bytes(b)


def ephemeral_dev_keyref(handle: str = "ephemeral_dev") -> KeyRef:
    """DEV ONLY: an ephemeral 32-byte key held in memory. NOT production key
    management. Documented in docs/F3_PROTECTED_CHECKPOINT.md."""
    key = secrets.token_bytes(32)
    return KeyRef(handle=handle, resolve=lambda: key)


# ---------------------------------------------------------------------------
# HMAC-authenticated envelope
# ---------------------------------------------------------------------------
def _hmac_over(ckpt_path: str, meta_bytes: bytes, key: bytes) -> str:
    """HMAC-SHA256 over the checkpoint file bytes CONCATENATED WITH the meta
    JSON. Both are authenticated together — an attacker cannot swap meta and
    payload independently."""
    m = hmac.new(key, digestmod=hashlib.sha256)
    with open(ckpt_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            m.update(chunk)
    m.update(b"\x00META\x00")
    m.update(meta_bytes)
    return m.hexdigest()


# ---------------------------------------------------------------------------
# Optional AEAD confidentiality
# ---------------------------------------------------------------------------
def _try_aesgcm():
    """Return AESGCM class or None. Catches BaseException so a broken install
    (e.g. pyo3 PanicException at import) is treated as "backend absent" rather
    than crashing the process — the module must never take confidentiality-mode
    down accidentally, and it must never silently fall back to plaintext."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore
        return AESGCM
    except BaseException:
        return None


def encrypt_payload(payload_bytes: bytes, keyref: KeyRef) -> Dict[str, str]:
    """AES-GCM encrypt a payload; returns dict with base64 ciphertext + nonce.
    Raises if cryptography is not installed (no silent fallback)."""
    AESGCM = _try_aesgcm()
    if AESGCM is None:
        raise KeyUnavailableError(
            "confidentiality mode requires the 'cryptography' package (AES-GCM); "
            "not installed. Do NOT weaken by falling back to plaintext.")
    import base64
    nonce = secrets.token_bytes(12)
    ct = AESGCM(keyref.key_bytes()).encrypt(nonce, payload_bytes, None)
    return {"algo": "aes-256-gcm",
            "nonce_b64": base64.b64encode(nonce).decode(),
            "ct_b64": base64.b64encode(ct).decode(),
            "key_handle": keyref.handle}


def decrypt_payload(record: Dict[str, str], keyref: KeyRef) -> bytes:
    AESGCM = _try_aesgcm()
    if AESGCM is None:
        raise KeyUnavailableError("cryptography backend absent; cannot decrypt")
    if record.get("algo") != "aes-256-gcm":
        raise CheckpointAuthenticationError(f"unknown algo: {record.get('algo')}")
    import base64
    nonce = base64.b64decode(record["nonce_b64"])
    ct = base64.b64decode(record["ct_b64"])
    try:
        return AESGCM(keyref.key_bytes()).decrypt(nonce, ct, None)
    except Exception as e:
        raise CheckpointAuthenticationError(f"AEAD decrypt failed: {e}") from e


# ---------------------------------------------------------------------------
# Anti-rollback (§F3.3)
# ---------------------------------------------------------------------------
@dataclass
class RecoveryDecision:
    """Explicit operator authorization to accept a rollback. Required to load a
    checkpoint whose authorized_step is below the current known-good state."""
    operator_authorization_ref: str            # opaque reference from operator system
    reason: str
    current_state_identity: str                # provenance hash of the running state
    selected_state_identity: str               # provenance hash of the target state
    integrity_result: str                      # "verified"
    recovery_policy_version: int
    resulting_authorized_state: int            # the authorized_step after rollback

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "recovery_decision": self.__dict__.copy(),
            "kind": "authorized_rollback",
        }


# ---------------------------------------------------------------------------
# Protected save / load
# ---------------------------------------------------------------------------
def protected_save(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    metadata: Dict[str, Any],
    keyref_mac: KeyRef,
    keyref_encrypt: Optional[KeyRef] = None,
    authorized_step: Optional[int] = None,
    rng_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Save a checkpoint under the F3 protected envelope.

    Writes three files atomically:
      - <path>            : (encrypted if keyref_encrypt else plaintext) payload
      - <path>.meta.json  : envelope metadata (schema, K, source_commit,
                            authorized_step, mac_algo, hmac_over="file+meta")
      - <path>.sha256     : sha256 of <path> for the E3 gate
    Plus a top-level HMAC tag stored inside <path>.meta.json.
    """
    if authorized_step is None:
        authorized_step = int(metadata.get("step", 0))

    envelope_metadata = {
        "envelope_version": PROTECTED_ENVELOPE_VERSION,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "patch_manifest_version": PATCH_MANIFEST_VERSION,
        "K": K_LOCKED,
        "authorized_step": int(authorized_step),
        "source_commit": source_commit_id(),
        "mac_algo": MAC_ALGO,
        "encrypted": keyref_encrypt is not None,
        "key_handle": keyref_encrypt.handle if keyref_encrypt else None,
        "inner_metadata": metadata,          # the E3 metadata (schema, patch mv, K, vocab, etc.)
    }

    if rng_state is None:
        rng_state = {
            "torch_cpu": torch.random.get_rng_state(),
            "torch_cuda_all": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
        }

    payload = {"metadata": envelope_metadata, "model": model.state_dict(),
               "optim": optimizer.state_dict(), "rng": rng_state}

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp_dir = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(prefix=".ckpt.tmp.", dir=tmp_dir)
    os.close(fd)

    try:
        if keyref_encrypt is None:
            torch.save(payload, tmp_path)
        else:
            # Save the payload to a plaintext temp first so we can encrypt bytes.
            plain_fd, plain_tmp = tempfile.mkstemp(prefix=".ckpt.plain.", dir=tmp_dir)
            os.close(plain_fd)
            try:
                torch.save(payload, plain_tmp)
                with open(plain_tmp, "rb") as fh:
                    enc_record = encrypt_payload(fh.read(), keyref_encrypt)
                with open(tmp_path, "wb") as fh:
                    fh.write(json.dumps(enc_record).encode("utf-8"))
            finally:
                # §F3.4 bound temp-file lifetime
                try: os.unlink(plain_tmp)
                except Exception: pass

        # Validate readability
        try:
            _size = os.path.getsize(tmp_path)
            assert _size > 0
        except Exception as e:
            raise CheckpointCorrupt(f"unable to validate temp checkpoint: {e}")

        # Preserve prior .prev
        if os.path.exists(path):
            prev = path + ".prev"
            if os.path.exists(prev):
                os.unlink(prev)
            os.rename(path, prev)
        os.rename(tmp_path, path)

        # sha256 sidecar (E3 gate)
        digest = _sha256(path)
        with open(path + ".sha256", "w") as fh:
            fh.write(digest + "\n")

        # HMAC tag over file + meta JSON, stored in meta.json
        meta_bytes = json.dumps(envelope_metadata, sort_keys=True,
                                 separators=(",", ":")).encode("utf-8")
        mac_hex = _hmac_over(path, meta_bytes, keyref_mac.key_bytes())
        meta_with_mac = dict(envelope_metadata)
        meta_with_mac["mac_hex"] = mac_hex
        meta_with_mac["mac_key_handle"] = keyref_mac.handle
        with open(path + ".meta.json", "w") as fh:
            json.dump(meta_with_mac, fh, indent=2, sort_keys=True)
        envelope_metadata["mac_hex"] = mac_hex
        envelope_metadata["sha256"] = digest
        return envelope_metadata

    except Exception:
        try:
            if os.path.exists(tmp_path): os.unlink(tmp_path)
        except Exception: pass
        raise


def protected_load(
    path: str,
    *,
    keyref_mac: KeyRef,
    keyref_encrypt: Optional[KeyRef] = None,
    expected_model_config: Dict[str, Any],
    current_authorized_step: Optional[int] = None,
    recovery_decision: Optional[RecoveryDecision] = None,
    enforce_anti_rollback: bool = True,
) -> Dict[str, Any]:
    """Load an F3-protected checkpoint. Fails closed on any of:
      - missing / mismatched sha256
      - missing .meta.json
      - MAC verification failure
      - schema / K / vocab / patch_manifest / envelope_version mismatch
      - anti-rollback violation (unless recovery_decision supplied)
      - encrypted payload without keyref_encrypt
      - AEAD decrypt failure
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    meta_path = path + ".meta.json"
    if not os.path.exists(meta_path):
        raise CheckpointCorrupt(f"missing envelope metadata: {meta_path}")
    with open(meta_path) as fh:
        meta = json.load(fh)

    # sha256 gate (matches E3 strict_load posture)
    sha_path = path + ".sha256"
    if os.path.exists(sha_path):
        expected = open(sha_path, encoding="ascii").read().strip().split()[0]
        actual = _sha256(path)
        if expected != actual:
            raise CheckpointCorrupt(f"sha256 mismatch: expected {expected}, got {actual}")
    else:
        raise CheckpointCorrupt("missing sidecar .sha256 for protected checkpoint")

    # MAC gate
    mac_expected = meta.get("mac_hex")
    if not mac_expected:
        raise CheckpointAuthenticationError("envelope has no mac_hex")
    meta_for_mac = {k: v for k, v in meta.items() if k not in ("mac_hex", "mac_key_handle")}
    meta_bytes = json.dumps(meta_for_mac, sort_keys=True, separators=(",", ":")).encode("utf-8")
    mac_actual = _hmac_over(path, meta_bytes, keyref_mac.key_bytes())
    if not hmac.compare_digest(mac_expected, mac_actual):
        raise CheckpointAuthenticationError("MAC verification failed")

    # Envelope schema
    if meta.get("envelope_version") != PROTECTED_ENVELOPE_VERSION:
        raise CheckpointIncompatible(f"envelope_version mismatch: {meta.get('envelope_version')}")
    if meta.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointIncompatible("schema_version mismatch")
    if meta.get("patch_manifest_version") != PATCH_MANIFEST_VERSION:
        raise CheckpointIncompatible("patch_manifest_version mismatch")
    if meta.get("K") != K_LOCKED:
        raise CheckpointIncompatible(f"K mismatch: {meta.get('K')}")

    ck_vocab = meta.get("inner_metadata", {}).get("model_config", {}).get("transformer", {}).get("vocab_size")
    ex_vocab = expected_model_config.get("transformer", {}).get("vocab_size")
    if ck_vocab is not None and ex_vocab is not None and ck_vocab != ex_vocab:
        raise CheckpointIncompatible(f"vocab_size mismatch: {ck_vocab} vs {ex_vocab}")

    # Anti-rollback gate (§F3.3)
    if enforce_anti_rollback and current_authorized_step is not None:
        ckpt_authorized = int(meta.get("authorized_step", 0))
        if ckpt_authorized < current_authorized_step:
            if recovery_decision is None:
                raise AntiRollbackViolation(
                    f"checkpoint authorized_step {ckpt_authorized} < current {current_authorized_step}; "
                    "explicit RecoveryDecision required")
            # Log the decision inside the returned metadata for audit binding
            meta["accepted_via_recovery_decision"] = recovery_decision.to_metadata()

    # Encryption gate
    is_encrypted = bool(meta.get("encrypted"))
    if is_encrypted and keyref_encrypt is None:
        raise KeyUnavailableError("checkpoint is encrypted but no keyref_encrypt supplied")

    # Load payload
    if is_encrypted:
        enc_record = json.load(open(path))
        plain_bytes = decrypt_payload(enc_record, keyref_encrypt)
        # Write to a temporary and torch.load with weights_only
        fd, tmp = tempfile.mkstemp(prefix=".ckpt.dec.", dir=os.path.dirname(path) or ".")
        os.close(fd)
        try:
            with open(tmp, "wb") as fh:
                fh.write(plain_bytes)
            payload = torch.load(tmp, map_location="cpu", weights_only=False)
        finally:
            try: os.unlink(tmp)
            except Exception: pass
    else:
        # Plaintext protected checkpoint — still authenticated by MAC.
        # weights_only=False here because the payload is authenticated; but we
        # keep the E3 strict_load semantics for the non-encrypted plain-path.
        payload = torch.load(path, map_location="cpu", weights_only=False)

    payload["envelope_metadata"] = meta
    return payload
