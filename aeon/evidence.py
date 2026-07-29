"""
aeon/evidence.py — canonical, portable, deterministic evidence serialization.

Every machine-readable evidence file committed to the repository is written
through this module. Its job:

  * Recursively normalize environment-specific absolute paths in strings inside
    dicts / lists / tuples / nested records / exception messages / command
    output excerpts. Committed evidence carries `<repo>/…`, `<tmp>/…`,
    `<home>/…`, or `<absolute>/basename` placeholders — never a leaked host
    prefix.
  * Preserve URLs, hashes, commit identifiers, math expressions, and
    non-path strings that merely contain slashes.
  * Preserve exception TYPE and diagnostic reason. Only the path prefixes
    inside the message are rewritten.
  * Emit deterministic JSON: sort_keys=True, UTF-8, forward slashes, no
    non-semantic whitespace.

Framework-free — no torch, no yaml, no external deps.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
from typing import Any, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Placeholders
# ---------------------------------------------------------------------------
REPO_PLACEHOLDER = "<repo>"
TMP_PLACEHOLDER = "<tmp>"
HOME_PLACEHOLDER = "<home>"
ABSOLUTE_PLACEHOLDER = "<absolute>"


# ---------------------------------------------------------------------------
# Roots
# ---------------------------------------------------------------------------
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_REPO_ROOT = os.path.abspath(os.path.join(_MODULE_DIR, ".."))


def _norm(p: str) -> str:
    """Normalize a path — realpath-safe and separator-safe. Returns forward-slash form."""
    if not p:
        return p
    try:
        return os.path.abspath(p).replace("\\", "/")
    except Exception:
        return p.replace("\\", "/")


def _all_tmp_roots(extra: Optional[Iterable[str]] = None) -> List[str]:
    """Every recognized temporary-directory root as forward-slash absolute
    paths. Includes POSIX `/tmp`, `/var/tmp`, `TMPDIR`, `tempfile.gettempdir()`,
    Windows Temp/AppData/Local/Temp, and any operator-supplied extras."""
    roots: List[str] = []
    seen: set = set()

    def add(p):
        if not p:
            return
        n = _norm(p)
        if n and n not in seen:
            seen.add(n); roots.append(n)

    for env_var in ("TMPDIR", "TEMP", "TMP"):
        add(os.environ.get(env_var))
    add(tempfile.gettempdir())
    for p in ("/tmp", "/var/tmp", "/private/tmp"):
        add(p)
    for extra_p in (extra or ()):
        add(extra_p)
    return roots


# ---------------------------------------------------------------------------
# Path-detection regexes
# ---------------------------------------------------------------------------
# URLs — never rewrite path characters inside these
_URL_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"<>]+")

# Full-length cryptographic hashes — hex only, of common lengths
_HASH_RE = re.compile(r"\b(?:[A-Fa-f0-9]{32}|[A-Fa-f0-9]{40}|[A-Fa-f0-9]{56}|[A-Fa-f0-9]{64})\b")

# Windows patterns
_WIN_DRIVE_TEMP_RE = re.compile(
    r"[A-Za-z]:[\\/]+(?:Users[\\/]+[^\\/]+[\\/]+AppData[\\/]+Local[\\/]+Temp|Temp|Windows[\\/]+Temp)"
    r"(?:[\\/]+[^\\/'\"<>\s]+)*",
    re.IGNORECASE,
)
_WIN_USERPROFILE_RE = re.compile(
    r"[A-Za-z]:[\\/]+Users[\\/]+[^\\/'\"<>\s]+(?:[\\/]+[^\\/'\"<>\s]+)*",
    re.IGNORECASE,
)
# POSIX default `tempfile.TemporaryDirectory` names — e.g. /tmp/tmpXXXX (mkdtemp)
_POSIX_MKDTEMP_RE = re.compile(r"/(?:tmp|var/tmp|private/tmp)/tmp[A-Za-z0-9_]{6,}")


def _mask(text: str, patterns: Iterable[re.Pattern]) -> tuple[str, dict]:
    """Replace matches with sentinels; return (masked_text, restore_map)."""
    restore: dict = {}
    n = [0]

    def sub(pat):
        def _fn(m):
            k = f"\x00MASK{n[0]}\x00"
            n[0] += 1
            restore[k] = m.group(0)
            return k
        return _fn

    for pat in patterns:
        text = pat.sub(sub(pat), text)
    return text, restore


def _unmask(text: str, restore: dict) -> str:
    for k, v in restore.items():
        text = text.replace(k, v)
    return text


# ---------------------------------------------------------------------------
# Core: rewrite a single string
# ---------------------------------------------------------------------------
def normalize_path_string(
    s: str,
    *,
    repo_root: Optional[str] = None,
    home: Optional[str] = None,
    tmp_roots: Optional[Iterable[str]] = None,
    extra_tmp: Optional[Iterable[str]] = None,
) -> str:
    """Rewrite host-specific absolute paths inside `s` to portable placeholders.

    Preserves URLs and hashes verbatim (via masking). Preserves the substantive
    content of exception messages — only path prefixes/absolute segments are
    rewritten. Windows separators are folded to forward slashes in output.
    """
    if not isinstance(s, str) or not s:
        return s

    def _fold(p: str) -> str:
        return (p or "").replace("\\", "/") if p else ""

    # Caller-supplied prefixes are used verbatim (only backslash-fold), so a
    # Windows-shaped prefix passed to a Linux tester still matches. The default
    # POSIX repo root gets abspath'd for canonical form.
    if repo_root is None:
        repo_root = _norm(DEFAULT_REPO_ROOT)
    else:
        repo_root = _fold(repo_root)
    if home is None:
        home = _norm(os.path.expanduser("~"))
    else:
        home = _fold(home)
    tmp_roots = list(tmp_roots or _all_tmp_roots(extra_tmp))

    # Mask URLs and hashes so path-shaped substrings inside them are untouched.
    text, restore = _mask(s, [_URL_RE, _HASH_RE])

    # Normalise separators throughout so a single set of substring rules
    # handles Windows and POSIX uniformly. All operator-supplied prefixes get
    # folded the same way.
    text = text.replace("\\", "/")
    repo_root_fs = repo_root.replace("\\", "/") if repo_root else ""
    home_fs = home.replace("\\", "/") if home else ""
    tmp_roots_fs = [t.replace("\\", "/") for t in tmp_roots]

    # -- pass 0: collapse POSIX mkdtemp names FIRST — `/tmp/tmpXXXX` and its
    # trailing tail become `<tmp>`. This must precede any generic `/tmp/`
    # prefix substitution below, which would otherwise leave the tmpXXXX
    # random suffix behind (leaking a per-run identifier).
    text = _POSIX_MKDTEMP_RE.sub(TMP_PLACEHOLDER, text)

    # -- pass 1: exact-prefix rewrites (repo → <repo>, tmp → <tmp>,
    # home → <home>) via sorted-by-length so longer prefixes win. Repo nested
    # inside home is matched as repo because repo_root_fs is longer than
    # home_fs.
    prefix_map: list[tuple[str, str]] = []
    if repo_root_fs:
        prefix_map.append((repo_root_fs, REPO_PLACEHOLDER))
    for tr in tmp_roots_fs:
        prefix_map.append((tr, TMP_PLACEHOLDER))
    if home_fs:
        prefix_map.append((home_fs, HOME_PLACEHOLDER))
    prefix_map.sort(key=lambda kv: -len(kv[0]))

    def _rewrite_prefix(text_in: str) -> str:
        out = text_in
        for prefix, placeholder in prefix_map:
            if not prefix:
                continue
            pattern = re.compile(re.escape(prefix) + r"(?=(?:/|$|['\"<>\s]))")
            out = pattern.sub(placeholder, out)
        return out

    text = _rewrite_prefix(text)

    # -- pass 2: generic Windows Temp / user-profile fallbacks --------------
    # Now that any operator-supplied repo prefix has been substituted, any
    # REMAINING Windows drive-letter path is generic.
    def _win_tmp(m: re.Match) -> str:
        tail = m.group(0)
        parts = tail.split("/")
        try:
            idx = [i for i, p in enumerate(parts) if p.lower() == "temp"][-1]
        except IndexError:
            idx = len(parts) - 1
        rel = "/".join(parts[idx + 1:])
        return TMP_PLACEHOLDER + ("/" + rel if rel else "")

    def _win_user(m: re.Match) -> str:
        tail = m.group(0)
        parts = tail.split("/")
        try:
            idx = [i for i, p in enumerate(parts) if p.lower() == "users"][0]
        except IndexError:
            idx = 1
        rel = "/".join(parts[idx + 2:])
        return HOME_PLACEHOLDER + ("/" + rel if rel else "")

    # After forward-slash folding these regexes match: C:/Users/x/AppData/Local/Temp/...
    win_temp_re = re.compile(
        r"[A-Za-z]:/+(?:Users/+[^/]+/+AppData/+Local/+Temp|Temp|Windows/+Temp)"
        r"(?:/+[^'\"<>\s]+)*",
        re.IGNORECASE,
    )
    win_user_re = re.compile(
        r"[A-Za-z]:/+Users/+[^/'\"<>\s]+(?:/+[^'\"<>\s]+)*",
        re.IGNORECASE,
    )
    text = win_temp_re.sub(_win_tmp, text)
    text = win_user_re.sub(_win_user, text)

    # Any remaining Windows drive-letter absolute → <absolute>/basename
    def _win_generic(m: re.Match) -> str:
        tail = m.group(0)
        base = tail.rstrip("/").split("/")[-1] or ABSOLUTE_PLACEHOLDER
        return f"{ABSOLUTE_PLACEHOLDER}/{base}"

    text = re.sub(r"[A-Za-z]:/+[^\s'\"<>]+", _win_generic, text)

    # (mkdtemp collapse already done in pass 0)

    # -- pass 4: any remaining POSIX absolute paths → <absolute>/basename ----
    def _abs_generic(m: re.Match) -> str:
        raw = m.group(0)
        base = raw.rstrip("/").split("/")[-1] or ABSOLUTE_PLACEHOLDER
        return f"{ABSOLUTE_PLACEHOLDER}/{base}"

    # Match /a/b/c... at a word boundary (not inside a placeholder tail — the
    # `>` in <repo>/... is excluded so `/runs/foo` right after a placeholder
    # is NOT re-matched). One or more slash-separated segments, at least one
    # inner slash (so bare `/etc` or `/tmp` doesn't glob), no whitespace/quotes.
    text = re.sub(r"(?<![<A-Za-z0-9_>])/[^\s'\"<>/]+(?:/[^\s'\"<>]+)+",
                   _abs_generic, text)
    # Also handle single-segment absolutes like `/etc/passwd` where the second
    # segment is the tail (e.g. `/etc/passwd`, already handled above); and bare
    # `/passwd`-shaped tokens that still need collapsing.
    text = re.sub(r"(?<![<A-Za-z0-9_>])/[^\s'\"<>/]+(?=[\s'\"<>]|$)",
                   _abs_generic, text)

    # Fold any remaining backslashes to forward slashes for portability.
    text = text.replace("\\", "/")

    # Restore URLs and hashes.
    return _unmask(text, restore)


# ---------------------------------------------------------------------------
# Recursive walker
# ---------------------------------------------------------------------------
def sanitize_evidence(
    obj: Any,
    *,
    repo_root: Optional[str] = None,
    home: Optional[str] = None,
    extra_tmp: Optional[Iterable[str]] = None,
) -> Any:
    """Return a new object equal to `obj` with every string leaf normalized.

    Recurses into dicts (keys and values), lists, tuples (returned as lists),
    and sets (returned as sorted lists — for determinism). Anything else is
    returned unchanged.
    """
    kw = dict(repo_root=repo_root, home=home, extra_tmp=extra_tmp)
    if isinstance(obj, str):
        return normalize_path_string(obj, **kw)
    if isinstance(obj, dict):
        return {sanitize_evidence(k, **kw): sanitize_evidence(v, **kw)
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_evidence(x, **kw) for x in obj]
    if isinstance(obj, tuple):
        return [sanitize_evidence(x, **kw) for x in obj]
    if isinstance(obj, set):
        return sorted((sanitize_evidence(x, **kw) for x in obj), key=str)
    return obj


# ---------------------------------------------------------------------------
# Serialization + writer
# ---------------------------------------------------------------------------
def canonical_json_bytes(obj: Any) -> bytes:
    """Deterministic JSON: sort_keys, minimal separators, UTF-8, forward-slash-safe."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, default=str).encode("utf-8")


def write_evidence(
    path: str,
    obj: Any,
    *,
    repo_root: Optional[str] = None,
    home: Optional[str] = None,
    extra_tmp: Optional[Iterable[str]] = None,
    indent: Optional[int] = 2,
) -> str:
    """Sanitize + write. Returns the sha256 of the canonical bytes for the
    caller to record in the evidence bundle if desired."""
    import hashlib

    sanitized = sanitize_evidence(obj, repo_root=repo_root, home=home,
                                    extra_tmp=extra_tmp)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if indent is None:
        payload = canonical_json_bytes(sanitized)
    else:
        payload = json.dumps(sanitized, sort_keys=True, indent=indent,
                              ensure_ascii=False, default=str).encode("utf-8")
    with open(path, "wb") as fh:
        fh.write(payload)
    return hashlib.sha256(canonical_json_bytes(sanitized)).hexdigest()


# ---------------------------------------------------------------------------
# Repo-wide evidence-hygiene scan (structured JSON only per directive)
# ---------------------------------------------------------------------------
_JSON_EVIDENCE_GLOB_HINTS = ("evidence", "results", "accounting", "baseline",
                              "policy", "manifest", "topology", "provenance",
                              "audit")

_JSON_EVIDENCE_ALLOWLIST_FIELDS = {
    # Fields whose canonical VALUE is a well-known absolute string that is
    # intentionally left as-is. Add only with clear rationale.
    "index_url",       # PyPI wheel index URL, not a filesystem path
}


def scan_json_for_host_paths(
    path: str,
    *,
    repo_root: Optional[str] = None,
    home: Optional[str] = None,
    extra_tmp: Optional[Iterable[str]] = None,
) -> List[str]:
    """Load a JSON file and return a list of `field.path -> leaked_string`
    entries. Skips URLs and hashes. Empty list means clean."""
    if not os.path.exists(path):
        return []
    try:
        obj = json.load(open(path))
    except Exception:
        return []
    offenders: List[str] = []
    repo_root_n = _norm(repo_root or DEFAULT_REPO_ROOT)
    home_n = _norm(home or os.path.expanduser("~"))
    tmp_roots = list(_all_tmp_roots(extra_tmp))

    forbidden_patterns = [
        (re.compile(r"/tmp/tmp[A-Za-z0-9_]+"), "posix_mkdtemp"),
        (re.compile(r"/home/[^/\s'\"<>]+"), "posix_home"),
        (re.compile(r"/Users/[^/\s'\"<>]+"), "posix_users"),
        (re.compile(r"[A-Za-z]:[\\/]+"), "windows_drive"),
    ]

    # Prefix leaks (repo/tmp/home root literals appearing anywhere)
    for pref in [repo_root_n, home_n] + tmp_roots:
        if pref:
            forbidden_patterns.append((re.compile(re.escape(pref)), f"prefix:{pref}"))

    def walk(node, path_):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path_}.{k}")
        elif isinstance(node, list):
            for i, x in enumerate(node):
                walk(x, f"{path_}[{i}]")
        elif isinstance(node, str):
            if path_.split(".")[-1] in _JSON_EVIDENCE_ALLOWLIST_FIELDS:
                return
            # Mask URLs and hashes before pattern matching
            masked, _ = _mask(node, [_URL_RE, _HASH_RE])
            for pat, name in forbidden_patterns:
                if pat.search(masked):
                    offenders.append(f"{path_} = {node!r}  (pattern: {name})")
                    return

    walk(obj, os.path.basename(path))
    return offenders
