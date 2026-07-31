"""aeon/job/generation.py — atomic per-generation checkpoint chain (W10-4).

Wraps ``aeon.protected_checkpoint.protected_save`` in a transactional
directory design so a crash between the payload write and the metadata
write cannot leave an authenticated-looking but incomplete generation.

Layout:

    <checkpoint_dir>/
        generation-000005/
            state.pt            (payload — HMAC-authenticated by meta.json)
            state.pt.sha256     (E3 gate)
            state.pt.meta.json  (envelope metadata + mac_hex)
            COMPLETE            (marker written last; contents = generation id)
        generation-000006.tmp/          (in-progress; NEVER selected by loaders)
            state.pt
            state.pt.sha256
            state.pt.meta.json
            (no COMPLETE)
        latest-authorized.txt           (single line: 'generation-000005')

Save flow (``generation_save``):

    1. mkdir generation-<step>.tmp/
    2. protected_save state.pt inside .tmp/
    3. verify meta.json + sha256 + MAC round-trip
    4. write COMPLETE (last)
    5. os.rename .tmp/ -> generation-<step>/ (single atomic step)
    6. os.replace latest-authorized.txt.tmp -> latest-authorized.txt

Any crash at steps 1-4 leaves .tmp/ behind — ``discard_incomplete`` cleans
those up on next call and no loader ever selects them. Step 5 is a single
atomic rename on POSIX and Windows. Step 6 uses os.replace which is atomic
per Python's documented guarantee.

Load flow (``latest_authorized_generation``):

    1. Read latest-authorized.txt (if present, that's the pointer).
    2. Otherwise, pick the highest-numbered generation-* directory that
       contains a COMPLETE marker.
    3. Call protected_load on state.pt under the caller's keyref.

Old ``ckpt_<N>.pt`` files from pre-W10-4 saves are NOT read by this
module. The launcher's resume enumeration understands both formats
during a transition window; new saves always go through generation_save.
"""
from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


COMPLETE_MARKER = "COMPLETE"
STATE_FILENAME = "state.pt"
LATEST_POINTER = "latest-authorized.txt"
GEN_PREFIX = "generation-"
GEN_STEP_WIDTH = 8  # generation-00000005


def generation_dir_name(step: int) -> str:
    return f"{GEN_PREFIX}{step:0{GEN_STEP_WIDTH}d}"


_GEN_RE = re.compile(r"^generation-(\d+)(\.tmp)?$")


def parse_generation_dir(name: str):
    """Return (step, is_tmp) or None if not a generation dir."""
    m = _GEN_RE.match(name)
    if not m:
        return None
    return int(m.group(1)), bool(m.group(2))


@dataclass
class Generation:
    step: int
    path: str  # absolute path to <checkpoint_dir>/generation-<step>/
    complete: bool
    state_path: str

    @property
    def is_incomplete(self) -> bool:
        return not self.complete


def list_generations(
    checkpoint_dir: str, *, include_incomplete: bool = False
) -> List[Generation]:
    """Return the generations under ``checkpoint_dir`` sorted highest-step
    first. If ``include_incomplete`` is False, ``.tmp`` directories and
    generation dirs without a COMPLETE marker are filtered out."""
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return []
    out: List[Generation] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        parsed = parse_generation_dir(child.name)
        if parsed is None:
            continue
        step, is_tmp = parsed
        complete = (child / COMPLETE_MARKER).exists() and not is_tmp
        if not include_incomplete and not complete:
            continue
        out.append(Generation(
            step=step, path=str(child), complete=complete,
            state_path=str(child / STATE_FILENAME)))
    out.sort(key=lambda g: g.step, reverse=True)
    return out


def discard_incomplete(checkpoint_dir: str) -> List[str]:
    """Delete every ``generation-*.tmp/`` and every ``generation-*/`` without
    a COMPLETE marker under ``checkpoint_dir``. Returns the paths removed
    (useful for logging). Safe to call on non-existent dir."""
    removed: List[str] = []
    root = Path(checkpoint_dir)
    if not root.is_dir():
        return removed
    for child in root.iterdir():
        if not child.is_dir():
            continue
        parsed = parse_generation_dir(child.name)
        if parsed is None:
            continue
        _, is_tmp = parsed
        if is_tmp or not (child / COMPLETE_MARKER).exists():
            shutil.rmtree(child)
            removed.append(str(child))
    return removed


def _pointer_path(checkpoint_dir: str) -> str:
    return os.path.join(checkpoint_dir, LATEST_POINTER)


def _update_latest_pointer(checkpoint_dir: str, gen_name: str) -> None:
    tp = _pointer_path(checkpoint_dir) + ".tmp"
    with open(tp, "w", encoding="utf-8") as fh:
        fh.write(gen_name + "\n")
    os.replace(tp, _pointer_path(checkpoint_dir))


def read_latest_pointer(checkpoint_dir: str) -> Optional[str]:
    p = _pointer_path(checkpoint_dir)
    if not os.path.exists(p):
        return None
    try:
        return open(p, encoding="utf-8").read().strip() or None
    except Exception:
        return None


def generation_save(
    checkpoint_dir: str,
    step: int,
    *,
    model,
    optimizer,
    metadata: Dict[str, Any],
    keyref,
    authorized_step: Optional[int] = None,
    rng_state: Optional[Dict[str, Any]] = None,
) -> Generation:
    """Save one atomic authenticated generation. See module docstring for
    the flow. Returns the ``Generation`` describing the promoted directory.

    Never mutates a previous generation. If the target step already exists
    as a complete generation, raises ``FileExistsError`` — atomic rotation
    means the caller must select a distinct step (typically ``step + 1``).
    """
    from aeon.protected_checkpoint import protected_save, protected_load

    root = Path(checkpoint_dir); root.mkdir(parents=True, exist_ok=True)
    target = root / generation_dir_name(step)
    if target.exists():
        raise FileExistsError(f"generation already exists: {target}")

    # Ensure no stale .tmp for the same step is lying around.
    tmp = root / (generation_dir_name(step) + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    state_path = tmp / STATE_FILENAME
    try:
        protected_save(str(state_path),
                        model=model, optimizer=optimizer, metadata=metadata,
                        keyref_mac=keyref,
                        authorized_step=authorized_step,
                        rng_state=rng_state)
        # Verify the round-trip inside the .tmp directory before promoting.
        # This is the invariant that makes .tmp "auditable but never
        # selected by loaders" — if verification fails, we discard and the
        # previous generation is untouched.
        _blob = protected_load(str(state_path),
                                keyref_mac=keyref,
                                expected_model_config=metadata.get(
                                    "model_config", {}))
        # COMPLETE marker is the last step before rename. Its contents are
        # human-readable state so an operator inspecting the tree knows what
        # was promoted.
        with open(tmp / COMPLETE_MARKER, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "generation": generation_dir_name(step),
                "step": int(step),
                "authorized_step": int(authorized_step or step),
                "mac_algo": _blob["envelope_metadata"]["mac_algo"],
                "envelope_version": _blob["envelope_metadata"]["envelope_version"],
            }, sort_keys=True))
        # Atomic promotion.
        os.rename(str(tmp), str(target))
    except Exception:
        # Best-effort cleanup so a partial .tmp does not accumulate.
        try:
            if tmp.exists():
                shutil.rmtree(tmp)
        except Exception:
            pass
        raise

    _update_latest_pointer(checkpoint_dir, generation_dir_name(step))
    return Generation(step=step, path=str(target), complete=True,
                       state_path=str(target / STATE_FILENAME))


def latest_authorized_generation(checkpoint_dir: str) -> Optional[Generation]:
    """Return the generation the pointer names, or the highest-step complete
    generation if the pointer is missing/absent. Never returns an incomplete
    generation."""
    ptr = read_latest_pointer(checkpoint_dir)
    if ptr:
        candidate = Path(checkpoint_dir) / ptr
        if candidate.is_dir() and (candidate / COMPLETE_MARKER).exists():
            parsed = parse_generation_dir(ptr)
            if parsed is not None:
                step, _ = parsed
                return Generation(step=step, path=str(candidate), complete=True,
                                   state_path=str(candidate / STATE_FILENAME))
    gens = list_generations(checkpoint_dir, include_incomplete=False)
    return gens[0] if gens else None


def previous_authorized_generation(
    checkpoint_dir: str, *, before_step: int
) -> Optional[Generation]:
    """Return the highest-step complete generation with ``step < before_step``.
    Used by Recovery to find the previous known-good state."""
    for g in list_generations(checkpoint_dir, include_incomplete=False):
        if g.step < before_step:
            return g
    return None
