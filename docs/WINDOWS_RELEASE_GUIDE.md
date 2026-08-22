# Windows Release Guide

This guide names the exact supported commands for producing and
verifying an Aeon Windows release from a clean repository checkout on
a Windows 10 or Windows 11 x64 machine with Python 3.11 x64
installed.

Every command is safe to re-run — the release script removes stale
outputs before rebuilding and fails closed at the first stage failure.

Prerequisites (installed once, per machine):

* Windows 10 or Windows 11 x64.
* Python 3.11 x64 (matches the pin in `pyproject.toml`).
* Inno Setup 6.x. Any of the following install locations works
  because `build_installer.ps1` discovers the compiler
  deterministically:
  1. Explicit `-InnoCompiler <path>` parameter.
  2. `$env:AEON_ISCC` environment variable.
  3. `ISCC.exe` on `PATH`.
  4. `%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe`.
  5. `%ProgramFiles%\Inno Setup 6\ISCC.exe`.
  6. `%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe` (per-user).
* The `packaging\windows\licenses\` directory populated per
  `packaging\windows\licenses\README.md`. The build refuses to
  proceed until every required upstream licence file is present.

Nothing here begins English training; nothing here downloads corpus
material; nothing here modifies the model, tokenizer, protected P2
checkpoint, or renderer. Startup verification is not English quality
verification — that separation is enforced by the launch script.

---

## 1. Build the bundle

```powershell
powershell.exe -ExecutionPolicy Bypass -File packaging\windows\build.ps1
```

Produces:

```
dist\Aeon\Aeon.exe
dist\Aeon\_internal\packaging\windows\RUNTIME_MANIFEST.json
dist\Aeon\_internal\packaging\windows\RUNTIME_MANIFEST.sha256
```

Runs the regression suite, populates the licences gate, generates the
manifest + sidecar, and executes `verify_bundle.py` against the built
bundle before the script exits.

## 2. Build the installer

```powershell
powershell.exe -ExecutionPolicy Bypass -File packaging\windows\build_installer.ps1
```

Or, if Inno Setup lives somewhere the discovery order does not cover:

```powershell
powershell.exe -ExecutionPolicy Bypass -File packaging\windows\build_installer.ps1 `
    -InnoCompiler 'C:\Path\To\ISCC.exe'
```

Produces:

```
dist\installer\AeonSetup.exe
dist\installer\AeonSetup.exe.sha256
```

Prints the resolved compiler path, the discovery source, the compiler
version banner, and the installer size + SHA-256.

## 3. Build the release ZIP (all of the above, in one command)

```powershell
powershell.exe -ExecutionPolicy Bypass -File packaging\windows\build_release.ps1
```

Runs stages 1, 2, and 3 above and additionally produces:

```
dist\Aeon_Desktop_7M_Research_Preview_Windows.zip
dist\Aeon_Desktop_7M_Research_Preview_Windows.zip.sha256
```

The ZIP contains only two entries: `AeonSetup.exe` and
`AeonSetup.exe.sha256`. The script verifies the ZIP entry list before
declaring success. On any failure it stops immediately; on success it
prints absolute paths, byte sizes, and lowercase SHA-256 hex for every
artefact. Pass `-InnoCompiler <path>` to forward to
`build_installer.ps1`. Pass `-OpenExplorer` to open the `dist\` folder
in File Explorer at the end — this is a convenience only; File
Explorer is never used as evidence.

## 4. Verify hashes

Every SHA-256 sidecar is a plain-text file whose first token is the
lowercase hex digest of the file whose name follows.

```powershell
Get-FileHash -Algorithm SHA256 dist\installer\AeonSetup.exe
Get-Content   dist\installer\AeonSetup.exe.sha256

Get-FileHash -Algorithm SHA256 dist\Aeon_Desktop_7M_Research_Preview_Windows.zip
Get-Content   dist\Aeon_Desktop_7M_Research_Preview_Windows.zip.sha256
```

If both digests match, the artefact matches what `build_release.ps1`
produced.

## 5. Launch the source bundle (startup only)

```powershell
powershell.exe -ExecutionPolicy Bypass -File packaging\windows\test_release.ps1
```

Runs:

```
dist\Aeon\Aeon.exe --version
dist\Aeon\Aeon.exe --verify-installation
```

Both must exit 0. To additionally launch the chat surface and confirm
it starts without exiting immediately:

```powershell
powershell.exe -ExecutionPolicy Bypass -File packaging\windows\test_release.ps1 -LaunchChat
```

The chat is started with `Aeon.exe --chat`, allowed to dwell for a
few seconds, and then closed cleanly. The script reports the process
ID and the executable path, and terminates any Aeon.exe it spawned
that would otherwise linger.

**This test proves startup only. It does not verify conversational
quality.** English behaviour evaluation is a separate program (see
`docs/en_train/EN_TRAIN_CORPUS_INTAKE_CONTRACT.md`) and is still
halted at `AWAITING_OFFLINE_CORPUS_SOURCES`.

## 6. Test an installed copy

Run the installer:

```powershell
.\dist\installer\AeonSetup.exe /VERYSILENT /SUPPRESSMSGBOXES
```

Then test the installed copy (usually at
`%LOCALAPPDATA%\Programs\Aeon`):

```powershell
powershell.exe -ExecutionPolicy Bypass `
    -File packaging\windows\test_release.ps1 `
    -BundleRoot "$env:LOCALAPPDATA\Programs\Aeon"
```

Or:

```powershell
powershell.exe -ExecutionPolicy Bypass `
    -File packaging\windows\test_release.ps1 `
    -BundleRoot "$env:LOCALAPPDATA\Programs\Aeon" `
    -LaunchChat
```

## 7. Locating outputs

```
dist\
├── Aeon\                                               # PyInstaller onedir bundle
│   ├── Aeon.exe
│   └── _internal\
│       └── packaging\windows\
│           ├── RUNTIME_MANIFEST.json
│           └── RUNTIME_MANIFEST.sha256
├── installer\
│   ├── AeonSetup.exe
│   └── AeonSetup.exe.sha256
├── Aeon_Desktop_7M_Research_Preview_Windows.zip
└── Aeon_Desktop_7M_Research_Preview_Windows.zip.sha256
```

## 8. Diagnosing missing Inno Setup

If `build_installer.ps1` (or `build_release.ps1`) reports Inno Setup
not found, the error message lists every path that was checked. Fix
one of:

* Install Inno Setup 6.x (per-user or system-wide) from
  <https://jrsoftware.org/isinfo.php>. The build never auto-downloads
  or auto-installs Inno Setup.
* Pass `-InnoCompiler 'C:\Path\To\ISCC.exe'` to
  `build_installer.ps1` or `build_release.ps1`.
* Set `$env:AEON_ISCC = 'C:\Path\To\ISCC.exe'` in the shell before
  invoking the script.

The observed Inno Setup 6.7.3 compiler banner prints
`Non-commercial use only`. This is recorded here so it is not missed;
it is a note for legal/licence review before any commercial
distribution and is not a legal conclusion.

## 9. Distinguishing successful startup from English-quality evaluation

The Windows release scripts verify:

* the bundle builds reproducibly,
* the runtime manifest hashes match the shipped files,
* `Aeon.exe --version` and `Aeon.exe --verify-installation` succeed,
* the packaging smoke test drives one worker step through the bundled
  tokenizer and a temporary throw-away corpus,
* (optionally) `Aeon.exe --chat` starts and remains alive for a short
  dwell.

They do **not** verify:

* the quality, coherence, or English fluency of any generated text,
* long-horizon conversational behaviour,
* claim ladders defined in `docs/en_train/…`,
* anything that requires an authored English corpus.

English behaviour evaluation belongs to the offline English training
program and is halted at `AWAITING_OFFLINE_CORPUS_SOURCES`. See
`docs/en_train/EN_TRAIN_CORPUS_INTAKE_CONTRACT.md`.

---

## Boundaries respected by these scripts

None of these commands modify:

* `aeon/hybrid.py`, `aeon/recursion.py`, `aeon/substrate/**`,
  `aeon/desktop/runtime.py`,
* `release-assets/aeon-desktop-p2-proxy/model/**`,
* `release-assets/aeon-desktop-p2-proxy/tokenizer/**`,
* the architecture fingerprint, `K = 16`, `MARGIN_H`, `MARGIN_C`,
  parameter count, state-dict keys or shapes,
* the protected P2 checkpoint or its SHA-256,
* the tokenizer file or vocabulary,
* generation parameters or English behaviour.

The packaging smoke worker (`verify_bundle.py`) constructs a
throw-away tiny model in a `TemporaryDirectory` for exactly one step
so the frozen bundle's data-source path is exercised end-to-end. It
is not authorization to train the release model and it does not touch
the protected checkpoint.
