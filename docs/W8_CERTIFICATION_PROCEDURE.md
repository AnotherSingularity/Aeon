# W8 — Clean-Machine Windows Certification Procedure

This document is the exact procedure a Windows CI runner or a controlled
local Windows host follows to convert the source in this branch into a
certified `AeonSetup.exe` (and the `Aeon.exe` it installs). The procedure
does **not** run on Linux — see §5 "External limitation" — but the entire
source tree needed to execute it is committed here and validated by
`tests/test_windows_packaging.py`.

The procedure MUST be executed on a clean Windows machine (fresh VM image,
no leftover Aeon installation, no environment overrides in the user
profile). Running it on a developer workstation where a previous Aeon
install already exists in `%LOCALAPPDATA%\Aeon` produces a signal-carrying
result but is **not** a certification.

## 1. Prerequisites (installed on the build host, not committed)

| # | requirement | notes |
|---|---|---|
| 1 | Windows 10/11 x64 | Server 2019+ acceptable |
| 2 | PowerShell 5.1+ or 7.x | 7.x preferred; scripts are `Set-StrictMode` clean |
| 3 | CPython 3.11.x (64-bit) | matches `packaging/windows/requirements-windows.lock` |
| 4 | Git for Windows | needed for `git rev-parse HEAD` in release metadata |
| 5 | Inno Setup 6.x | installer compiler; W6 script targets `%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe` |
| 6 | Windows SDK signtool.exe | must be discoverable via `AEON_SIGNTOOL_PATH` |
| 7 | Signing PFX (out-of-repo) | held in the build host's secret store; never committed |
| 8 | PFX password (out-of-repo) | read from secret store into `AEON_SIGN_CERT_PASS`; never logged |
| 9 | Network access to `pip install` from PyPI | one-time during the build; the produced installer requires no network |
| 10 | ~4 GB free disk in the build directory | PyInstaller onedir output + Inno compressed output |

Nothing above is committed to the repository. The build host provides them.

## 2. Certification environment variables

These are set **outside** this repository, in the build host's session.
The scripts read them; they are never printed by name-value into a log
that leaves the machine.

```
AEON_SIGNTOOL_PATH   = C:\Program Files (x86)\Windows Kits\10\bin\<ver>\x64\signtool.exe
AEON_SIGN_CERT_PFX   = <path to PFX held in the build host secret store>
AEON_SIGN_CERT_PASS  = <retrieved from secret store>   ; never echoed
AEON_SIGN_TIMESTAMP  = http://timestamp.digicert.com   ; RFC 3161
```

If `AEON_SIGN_CERT_PASS` is unset, `sign.ps1` throws before invoking
signtool. The signing step is optional at build time (an unsigned build
is a signalled development artifact); `release_metadata.py --signed`
records the signing intent in `RELEASE.json`.

## 3. Reproducible build sequence

Every command is executed from the repository root of a fresh checkout of
this branch at the tip commit.

### 3.1 Regression baseline (Windows-side confirmation)

```
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --disable-pip-version-check -r packaging\windows\requirements-windows.lock
python -m pip install pyinstaller==6.11.1
python -m pytest tests\ -x -q     # must be all-green
```

The regression suite includes `tests/test_windows_packaging.py`, which
re-validates the packaging source structure on Windows itself.

### 3.2 Freeze release metadata

```
python packaging\windows\release_metadata.py `
    --out packaging\windows\RELEASE.json `
    --semantic-version 0.2.3 `
    --build-type certified `
    --signed
```

Writes `RELEASE.json` with `semantic_version`, `source_commit` (from
`git rev-parse HEAD`), `build_type`, and `signed: true`. The `--signed`
flag is a **claim** written to metadata; the actual cryptographic signing
happens in §3.5.

### 3.3 PyInstaller onedir bundle

```
.\packaging\windows\build.ps1
```

`build.ps1` runs `pyinstaller packaging\windows\Aeon.spec` and produces
`dist\Aeon\Aeon.exe` with the full onedir payload. It then runs
`verify_bundle.py`, which walks the bundle and asserts:

* `Aeon.exe` present.
* No `.pfx`, `.key`, `.pem`, `.p12`, `.crt`, `.pass` files present.
* No test directories under `dist\Aeon\`.
* No CUDA runtime DLLs (`cudart*`, `cudnn*`, `nvcuda*`).
* `python*.dll` present at the bundle root.

### 3.4 Runtime manifest

```
python packaging\windows\generate_runtime_manifest.py `
    --bundle dist\Aeon `
    --release packaging\windows\RELEASE.json
```

Walks `dist\Aeon\` deterministically and produces
`dist\Aeon\packaging\windows\RUNTIME_MANIFEST.json`:
`{ files: [ {path, size, sha256}, ... ], release: {…} }`. This is the
manifest that `aeon.integrity.verify_installed_manifest()` consumes at
first launch and after every upgrade.

### 3.5 Sign `Aeon.exe`

```
.\packaging\windows\sign.ps1 -AeonExe dist\Aeon\Aeon.exe -Installer dist\installer\AeonSetup.exe
```

`sign.ps1` invokes `signtool sign /fd sha256 /td sha256 /tr $ts /f $pfx /p $pass`
on both paths. The installer path may not exist yet at this call site —
the script's order is Aeon.exe first (must exist), then installer (built
in §3.6). The recommended sequence is to sign only `Aeon.exe` here
(`sign.ps1` becomes a two-step call: after §3.4 sign the bundled EXE, then
after §3.6 sign the installer). Both signatures use the same PFX and the
same RFC 3161 timestamp URL.

### 3.6 Build the installer

```
.\packaging\windows\build_installer.ps1
```

Runs `ISCC.exe packaging\windows\AeonInstaller.iss`. Output:
`dist\installer\AeonSetup.exe` + `dist\installer\AeonSetup.exe.sha256`.
The installer script refuses to compile if `RUNTIME_MANIFEST.json` is
missing under the bundle root (Inno `PrepareToInstall` check).

### 3.7 Sign `AeonSetup.exe`

Re-run `sign.ps1` (or its second invocation) to sign the installer with
the same PFX / timestamp URL. The `signtool verify /pa /v` inside the
script confirms the chain.

### 3.8 Record final measurements

Copy `dist\installer\AeonSetup.exe.sha256` and the newly-embedded
`RELEASE.json` (extractable from the installed bundle) into
`docs\w9_certification_evidence.json` (see W9). The build host records:

* `AeonSetup.exe` size in bytes.
* `AeonSetup.exe` SHA-256.
* Wall-clock build time on the Windows host.
* Signtool verification output (chain path, not the PFX).

## 4. Post-build clean-machine validation

Perform each check on a **different**, clean Windows machine (or a fresh
Windows VM snapshot). This isolates the installer from the build host's
state.

| # | check | pass condition |
|---|---|---|
| 1 | Right-click `AeonSetup.exe` → Properties → Digital Signatures | signature present, timestamp present, chain to a trusted root |
| 2 | Double-click `AeonSetup.exe` | UAC prompt does NOT elevate; install proceeds without admin |
| 3 | Install path | `%LOCALAPPDATA%\Programs\Aeon\` populated; `Aeon.exe` present |
| 4 | User data path | `%LOCALAPPDATA%\Aeon\` created on first launch |
| 5 | Launch `Aeon.exe` (double-click) | no console window appears; Tkinter launcher opens |
| 6 | Launcher → Installation panel | reports "Installation valid" (`aeon.integrity.verify_installed_manifest` PASS) |
| 7 | Launcher → Config wizard | writes `%LOCALAPPDATA%\Aeon\config\user_config.json` under user-writable path |
| 8 | Launcher → Preflight | 17-check report, verdict READY or READY_WITH_WARNINGS |
| 9 | Launcher → Start training | worker process spawns detached (no console); status transitions `STARTING → PREFLIGHT → RUNNING` |
| 10 | Launcher → Stop safely | worker transitions `RUNNING → STOP_REQUESTED → CHECKPOINTING → STOPPED`; checkpoint file present under `%LOCALAPPDATA%\Aeon\checkpoints\` |
| 11 | Close launcher while training | worker keeps running (independent process); reopening launcher reattaches |
| 12 | Kill worker via Task Manager, reopen launcher | previous job marked `RECOVERY_REQUIRED`; launcher offers recovery |
| 13 | Reboot host while training | on reboot, launcher reports the prior job as `RECOVERY_REQUIRED` (worker didn't unlock cleanly) |
| 14 | Tamper `%LOCALAPPDATA%\Programs\Aeon\Aeon.exe` (append a byte) | launcher reports "Installation invalid — file mismatch" via `verify_installed_manifest` |
| 15 | Upgrade over previous version while a checkpoint is being written | installer refuses with `MB_ERROR` ("Aeon is currently saving a checkpoint…") |
| 16 | Uninstall via Settings → Apps | uninstaller runs without admin; asks about deleting user data with **No** as default; leaving user data preserved keeps `%LOCALAPPDATA%\Aeon\` intact |
| 17 | Reinstall after uninstall-with-preserved-data | prior checkpoints and config still present; launcher offers resume |
| 18 | Windows Defender / SmartScreen | signed installer does not trigger SmartScreen "Unrecognized app" (or, if it does on a fresh reputation, "More info → Run anyway" is honoured; SmartScreen reputation is a Microsoft-side metric, not a code defect) |

## 5. External limitation (reiterated)

This Linux container **cannot** execute PyInstaller for Windows, cannot
run Inno Setup's ISCC, and has no access to a Windows SDK signtool or
signing PFX. The procedure above is authoritative but requires an
authorised Windows execution environment to produce the certified
artifacts. The steps that are **verifiable in this session** — spec
structure, hook contents, ISS structure, sign.ps1 env-var-only usage,
manifest generator round-trip, no secrets in any packaging file, no
absolute host paths in any packaging file — are locked by
`tests/test_windows_packaging.py` (13 checks, all passing).

The certified `AeonSetup.exe` size, SHA-256, and signature chain are
therefore recorded by the Windows CI runner into
`docs/w9_certification_evidence.json` **when the procedure above is
executed on Windows**, and that JSON is committed on top of this branch
by whoever runs the certification.

## 6. Signing key hygiene (§W7)

None of the following ever enters the repository:

* PFX file bytes (`.pfx`, `.p12`).
* Private-key material (`.key`, `.pem` for private keys).
* PFX passwords, cert-store passwords, HSM PINs.
* Timestamp-server API keys (public HTTP timestamp URLs are fine).

`tests/test_windows_packaging.py::test_no_signing_material_committed_anywhere`
walks the packaging tree on every test run and asserts none of the above
extensions exist. `test_sign_ps1_uses_env_vars_only` asserts the signing
script references credentials **only** through environment variables and
has no hard-coded password hints.

Loss of the signing PFX is a build-host problem, not a code problem. Its
recovery is the responsibility of whoever operates the Windows secret
store, not this repository. Aeon's runtime integrity is independent of
the Authenticode chain — it uses `RUNTIME_MANIFEST.json` (SHA-256 per
file) for install verification. Authenticode signs the delivery vector
(installer + EXE), not Aeon's execution model.
