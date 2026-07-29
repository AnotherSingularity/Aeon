# W0 — Windows Packaging Inheritance Audit

**Starting commit:** `0d7bbb9` (F9.1 close-out).
**New branch:** `claude/windows-packaging`, based on `0d7bbb9`. No E/F-series history rewrite.
**Working environment for this session:** Linux x86_64 container (kernel 6.18.5). See §External limitation below.

## 1. Handoff verification

| # | check | result |
|---|---|---|
| 1 | Local branch | `claude/windows-packaging` |
| 2 | Local HEAD | `0d7bbb9` (matches directive baseline) |
| 3 | Remote origin/claude/funny-cori-a3k5cf | `0d7bbb9` (F9.1 tip) |
| 4 | Working tree | clean |
| 5 | Additive history preserved | E0→F9→F9.1 chain intact, no rewrite |
| 6 | Inherited regression baseline | 155/155 (see `docs/w0_baseline.txt`) |
| 7 | Python | 3.11.15 |
| 8 | PyTorch | 2.13.0+cu130 (container fallback; primary pin torch==2.5.1) |
| 9 | PyInstaller | **not installed in this Linux session** (would target Windows for W5) |
| 10 | Inno Setup | **not installed / not applicable on Linux** (Windows-only tool for W6) |
| 11 | Target Windows architecture | x64 (first certified target) |
| 12 | Current parameter accounting | 350.28 M trainable (unchanged — `docs/e6_parameter_accounting.json`) |
| 13 | Architecture manifest identity | see `docs/PRESERVATION_MANIFEST.md` (E-series, unchanged) |
| 14 | Security policy identity | `docs/SECURITY_MODEL.md` + `docs/runtime_policy.json` (F4) |
| 15 | Runtime policy identity | `docs/runtime_policy.json` (F4), sha in `docs/f9_final_evidence.json::artefact_hashes` |

## 2. Windows-compatibility gap report

The E/F-series was written for POSIX. Every gap below must be addressed in the W-series without weakening any inherited invariant.

### Gap category A — path separators / conventions

| gap | file(s) | current behaviour | remediation phase |
|---|---|---|---|
| Hard-coded `/tmp` | `aeon/checkpoint.py`, `aeon/protected_checkpoint.py`, `aeon/runtime_policy.py` (template `<tmp>`), `scripts/e5_certify.py`, `scripts/f7_certify.py`, `scripts/f8_recovery.py` | POSIX `/tmp` roots | Already tolerated by F9.1 sanitizer (`_all_tmp_roots` includes `TEMP`/`TMP` env vars + Windows Temp patterns). W1 substitutions map `<tmp>` to `%TEMP%` at runtime. |
| `os.path.join` — fine | many files | already OS-agnostic | none |
| `/home/user/AeonV0.02` mentioned | docs `F0_INHERITANCE_AUDIT.md` (historical), `F9_DEFINITION_OF_DONE.md` (row 29 documentation) | intentional documentation references | none — these describe the F0 fix, not live paths |

### Gap category B — process model

| gap | current | remediation |
|---|---|---|
| No fork-based multiprocessing | Aeon uses no `multiprocessing` at runtime (verified: `grep -R "multiprocessing" aeon/ scripts/`) | none needed |
| POSIX signals | not used (Aeon has no signal handlers) | W3 uses graceful `stop.request` file protocol — cross-platform |
| Bash launcher | scripts under `scripts/` are Python `.py` files; only `packaging/*.ps1` will be shell (Windows-side) | W5 build script is PowerShell (Windows-native) |
| Executable `.sh` files | none present | none |

### Gap category C — file locking / resource controls

| gap | current | remediation |
|---|---|---|
| Linux-only `fcntl` locking | `grep -R "fcntl" aeon/` returns zero — Aeon does not use `fcntl` | W3 single-instance lock uses `msvcrt.locking` on Windows and file-existence on POSIX; both cross-platform |
| Linux-only cgroups | Referenced only in `SECURITY_MODEL.md` as "deployment work" — not enforced in code | none |
| Linux-only `resource` module | not used in `aeon/` runtime path | none |

### Gap category D — packaging

| gap | current | remediation phase |
|---|---|---|
| No frozen entry point | `scripts/train.py`, `scripts/infer.py`, `scripts/diagnose.py` are ordinary Python scripts | W1 unified `aeon/entry.py` dispatcher |
| No launcher | ops assumed CLI | W2 Tkinter launcher (stdlib) |
| No worker lifecycle | training is foreground | W3 detached worker with job dir |
| No installer | none | W5 PyInstaller spec + W6 Inno Setup script |
| No signed executable | none | W7 signing pipeline (source-only in this branch) |
| No LOCALAPPDATA convention | configs live in `runs/` under repo | W4 first-run wizard writes to `%LOCALAPPDATA%\Aeon\config` on Windows |

### Gap category E — GUI / display

| gap | current | remediation |
|---|---|---|
| No GUI | none | W2 uses `tkinter` from Python stdlib (already ships with Windows Python installers and with a PyInstaller bundle by default) |
| Console visibility | ordinary Python | W1/W2 mark `Aeon.exe` as GUI subsystem (`--windowed` PyInstaller) — no console appears |

## 3. Inherited invariants — untouched by W-series

The W-series is **packaging only**. Every architectural invariant listed in §3 of the directive stays exactly as certified in the E/F-series (see `docs/PRESERVATION_MANIFEST.md`, `docs/DEFINITION_OF_DONE.md`, `docs/F9_DEFINITION_OF_DONE.md`). The full regression suite (155 tests) will be rerun at each W-phase and at W9 close-out; any packaging change that breaks an inherited test blocks the W-phase.

## 4. External limitation (structural)

**This Linux container cannot produce a certified Windows build.** The directive itself names this rule in §14: *"Because the Windows executable must be built on Windows, use either: An authorized local Windows build host, or A controlled Windows CI runner. Do not treat a Linux-produced artifact as a certified Windows build."*

**Scope split for this branch:**

| phase | Linux-buildable here | Requires Windows |
|---|:-:|:-:|
| W0 audit | ✓ | — |
| W1 unified entry point (Python source + tests) | ✓ | — |
| W2 launcher (Tkinter source + headless tests) | ✓ | Visual/manual GUI test |
| W3 worker lifecycle (job dir + safe-stop protocol source + tests) | ✓ | Windows process detachment test |
| W4 config + preflight (source + tests) | ✓ | — |
| W5 PyInstaller `.spec` + `runtime_hook.py` + `verify_bundle.py` (source) | ✓ (source) | PyInstaller build → `Aeon.exe` |
| W5 build.ps1 (source) | ✓ (source) | Actually run on Windows |
| W6 `AeonInstaller.iss` + `build_installer.ps1` (source) | ✓ (source) | Actually build `AeonSetup.exe` |
| W7 `sign.ps1` + release_metadata (source; NO keys committed) | ✓ (source) | Actual SignTool signing |
| W8 clean-machine certification | — | ✓ full Windows procedure |
| W9 closure | ✓ (documentation) | Recording final measurements from Windows CI |

I will produce **every source deliverable** so a Windows CI runner or local Windows host can execute the build end-to-end without further code changes. W8 will be a concrete procedure document with commands. W9 will explicitly enumerate the Windows-only gates that require signoff outside this session.

## 5. W0 exit gate

- [x] Repository at `0d7bbb9`, branch `claude/windows-packaging`, working tree clean.
- [x] Inherited regression baseline recorded (155/155 — see `docs/w0_baseline.txt`).
- [x] Environment versions recorded (Python 3.11.15, torch 2.13.0 in this session; primary pin torch 2.5.1).
- [x] Windows-compatibility gap report complete with per-gap remediation phase.
- [x] External limitation (no Windows build environment here) documented explicitly with reduced-scope path.

**W0 exit gate: PASS.**
