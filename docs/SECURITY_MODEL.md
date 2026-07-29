# Aeon V0.02.03 — Local Security Model

Covers §10.4 local-security requirements. Aeon runs in a laptop-class environment
without implicit authority to reach the network, execute shell commands, or
touch the filesystem outside its own paths.

## 1. What the code assumes about its environment

- **No network authority.** Aeon downloads nothing at runtime. Tokenizer + corpus
  + model config paths are set locally. `pyproject.toml` pins `torch==2.5.1` etc.
- **No shell execution authority.** No `subprocess.run` on user input; the only
  subprocess calls are `aeon.checkpoint.source_commit_id` (harmless read-only
  `git rev-parse HEAD`) and the E5 certification runner spawning `python
  scripts/diagnose.py` on a known local path.
- **No filesystem write outside allowlisted paths.** `train.py` writes only
  under `train.out_dir`; `diagnose.py` writes only next to the source checkpoint
  or the `--out` path. Neither traverses outside their configured roots.
- **No `eval` on arbitrary serialized objects.** `aeon.checkpoint.strict_load`
  calls `torch.load(..., weights_only=True)` — the untrusted-pickle path is
  refused with a named `CheckpointIncompatible` (never silently downgraded).

## 2. Untrusted inputs — treatment

| input | trust posture | mitigation |
|---|---|---|
| Corpus text | untrusted | Only tokenized; never `exec`-ed or `eval`-ed. Tokenizer treats bytes as bytes (SentencePiece + byte-fallback). |
| Corpus file paths | trusted (operator sets them) | Refused if outside `data.corpus` root at read time. |
| Tokenizer `.model` | trusted (operator sets identity) | Path pinned in checkpoint metadata; strict_load refuses a swap. |
| Checkpoint `.pt` on disk | UNTRUSTED for a shared machine | `weights_only=True`, sha256 sidecar, metadata gate (§10.4). |
| Config YAML | trusted | Config-invariant tests refuse silent K / margin / adaptive-clock drift. |

## 3. `strict_load` fail-closed behaviour

`aeon/checkpoint.py::strict_load` rejects, without loading state, on any of:

1. Missing sidecar `.sha256` (when `require_sha256=True`, the default).
2. Sidecar sha256 mismatch (bytes tampered).
3. `weights_only=True` not supported by the installed torch (documented refusal
   pointing to the docs; do not downgrade the security posture).
4. Metadata missing / not a dict.
5. `schema_version` mismatch.
6. `patch_manifest_version` mismatch (the six-patch topology has changed;
   resume would silently reintroduce a fixed defect).
7. `K` != 16 (§3.4 lock).
8. `transformer.vocab_size` mismatch (a silent tokenizer swap would corrupt
   generation).

Tested in `tests/test_checkpoint.py` (`test_reject_incompatible_metadata_*`,
`test_reject_corrupt_sha256`, `test_missing_sha256_rejected_when_required`,
`test_strict_load_uses_weights_only_or_hardened`).

## 4. Atomic save — data safety

`aeon/checkpoint.py::atomic_save` guarantees:

1. **Prior checkpoint retention.** If `path` already exists, it is renamed to
   `path + ".prev"` BEFORE the new file is renamed into place.
2. **Interruption survival.** A crash during `torch.save` leaves a temp file
   that is cleaned on the next call — never overwrites the prior checkpoint.
3. **Integrity metadata.** A sidecar `.sha256` file is written AFTER the atomic
   rename; on load, the sidecar's hash is checked against the file's current
   contents.

Tested in `tests/test_checkpoint.py::test_atomic_save_survives_interrupted_write`.

## 5. Credentials

Aeon does not read any credentials. Nothing in `aeon/` or `scripts/` touches
`~/.aws`, `~/.config`, `HF_TOKEN`, etc.

## 6. Environment-variable posture

Aeon reads no environment variables at runtime. The Python entry points take
their entire configuration from the YAML file plus CLI flags.

## 7. Logging

`aeon.observability.JsonlWriter` guards every write in `try/except`. A write
failure marks the writer broken but never raises through to the training loop —
this preserves the primary run when disk pressure hits, and it is intentional
(directive §8.5).
