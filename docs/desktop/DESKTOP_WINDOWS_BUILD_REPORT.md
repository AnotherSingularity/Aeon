# DESKTOP-WINDOWS — Windows Build / Install / Acceptance Report

**Status:** **STATE B — Windows execution required.**

The reconciliation branch has proven every Linux/headless requirement
that R2–R5 could verify. Windows-native PyInstaller freeze + Inno
installer + clean-install acceptance + restart + upgrade + uninstall
require a Windows environment and cannot be executed from this Linux
container. This report enumerates each Windows gate and the exact
next command.

---

## WINDOWS-0 — Native pre-flight

**Not executable from this container** (Linux only). On a Windows
runner (or workstation) with Python 3.11, PowerShell, PyTorch CPU,
PyInstaller, and Inno Setup 6.x installed:

```powershell
git fetch origin --prune
git switch claude/aeon-desktop-7m-validation
git pull --ff-only origin claude/aeon-desktop-7m-validation
git status --short
git rev-parse HEAD                                     # expect the reconciliation head

python scripts/export_aeon_desktop_model.py             # rebuild release bundle
Get-FileHash release-assets\aeon-desktop-p2-proxy\model\aeon-p2-proxy-inference.pt -Algorithm SHA256
# Expect: c10350ac5569cd44e93226b40b1aa4cd0b8b2773ebe45401719946038015f1e4
```

If the hash mismatches, correct the exporter to separate deterministic
model bytes from variable build metadata (torch's `pickle` protocol
already produces deterministic bytes when `_use_new_zipfile_serialization=True`
and same-precision same-order tensors are written — no environmental
non-determinism is expected). Report the mismatch, do not proceed to
WINDOWS-1.

Then run the Windows-compatible regression:
```powershell
foreach ($t in Get-ChildItem tests\test_*.py) { python $t.FullName }
```
Expect: same test totals as recorded in the release evidence,
minus any tests that require Linux-only primitives (`resource.getrusage`
is available on both Windows and POSIX under `RUSAGE_SELF`; the soak
runner uses that call — if `ru_maxrss` returns bytes on Windows vs
KiB on Linux, adjust the divisor accordingly).

---

## WINDOWS-1 — Frozen runtime

Command:
```powershell
powershell.exe -ExecutionPolicy Bypass -File packaging\windows\build.ps1
```

Expected outputs:
* `dist\Aeon\Aeon.exe`
* `dist\Aeon\_internal\` (all bundled modules + DLLs)
* `dist\Aeon\_internal\release-assets\aeon-desktop-p2-proxy\` (release bundle)

Acceptance:
* Build succeeds without errors.
* `dist\Aeon\Aeon.exe --version` prints the release metadata.
* `dist\Aeon\Aeon.exe --verify-installation` returns 0.
* `dist\Aeon\Aeon.exe --chat` launches the chat window (no console).
* Copy `dist\Aeon\` to a directory OUTSIDE the repo (e.g. `C:\Temp\Aeon\`),
  cd out of the repo, and verify `Aeon.exe --chat` still works.
* Model, tokenizer, and manifests resolve via
  `aeon.windows_paths.installed_resource_root()` — not CWD.
* Settings write under `%LOCALAPPDATA%\Aeon` (or the app-data path
  configured by `aeon.windows_paths.user_data_root()`).

Reject if:
* Training corpus paths (`research-data\`) are present in `dist\Aeon\`.
* Optimizer state is present in the exported model file.
* Sealed-test text is present.
* Development scripts (`scripts\train.py`, `scripts\run_l3_l4_l5.py`,
  `scripts\run_pipeline_stage.py`, `scripts\acis_workload_certify.py`)
  are present.

Do not proceed to WINDOWS-2 until the copy-out-of-repo test passes.

---

## WINDOWS-2 — Installer

Command:
```powershell
powershell.exe -ExecutionPolicy Bypass -File packaging\windows\build_installer.ps1
```

Expected output: `dist\AeonSetup.exe`.

Record:
* installer path
* installer SHA-256 (`Get-FileHash dist\AeonSetup.exe -Algorithm SHA256`)
* installer size in bytes
* file version + product version (`(Get-Item dist\AeonSetup.exe).VersionInfo`)
* release identity (matches `manifests\release_manifest.json.release_id`)

Reject inclusion of:
* `research-data\incoming\`
* `research-data\AEON-LBC-1\processed\`
* Sealed test bytes
* `runs\aeon_lbc1_P2\final.pt` (training checkpoint with optimizer)
* `docs\latent_bypass\l3_reaction_coordinate_evidence.json` /
  `l4_telemetry_evidence.json` / `l5_causal_evidence.json`
* `.git`, `__pycache__`, `.pytest_cache`
* `tests\` (excluded via `AeonInstaller.iss` `[Files]` scope)
* `packaging\windows\*.ps1` (build-time tools; not runtime)

Unsigned installer is acceptable (§33 authorizes this for the research
preview). Report `unsigned = true` in the evidence; do not fabricate a
signing claim.

---

## WINDOWS-3 — Clean-install vertical slice

Requires a clean Windows user profile or clean VM snapshot with no
existing Aeon installation.

```
1. Copy AeonSetup.exe to the clean profile.
2. Right-click -> Run as normal user (installer is per-user or
   HKCU-scoped for the research preview).
3. Step through the wizard; accept default install path.
4. Launcher via Start Menu shortcut "Aeon Chat" (WINDOWS-2 will
   have added this shortcut to Inno's [Icons] block).
5. Wait for READY status.
6. Type "Once upon a time" into the prompt, click Send.
7. Verify authentic streamed output (Aeon token IDs decode via the
   bundled tokenizer).
8. Submit a second, longer prompt with max_new_tokens=64.
9. Click Stop; verify state returns to READY within ~200ms + one
   inter-token interval; verify partial text is preserved and
   marked [cancelled].
10. Click New Session; verify transcript clears and a new session
    id appears in diagnostics.
11. Type another prompt; verify generation works.
12. Click Clear Conversation; verify token_history resets.
13. Close the window.
14. Verify no orphan process remains (Get-Process aeon | Should be $null).
```

Capture:
* Sanitized screenshot on READY
* Sanitized screenshot mid-generation
* Sanitized screenshot after Stop
* Structured evidence: `docs\desktop\desktop_windows_evidence.json`

No console window may appear during any step. No internet is
required. No Git or system Python required.

---

## WINDOWS-4 — Restart, recovery, upgrade, uninstall

Repeat launch → generate → close → verify-exit → relaunch → generate
ten times. Record process ids and exit codes.

Runtime crash injection: launch `Aeon.exe --chat`, use Task Manager
to kill the running Aeon.exe mid-generation. Expected user-facing
behavior: the shell dies. That is the accepted trade-off of the
in-process design per DESKTOP-R1's DESKTOP_AUTHORITATIVE_PATH.md
`in_process_runtime_justification.trade_off_acknowledged`. Restart
the shell manually; verify it starts cleanly.

Invalid release: modify installed
`_internal\release-assets\aeon-desktop-p2-proxy\model\aeon-p2-proxy-inference.pt`
by flipping a byte; run `Aeon.exe --chat`; expect `MODEL_DIGEST_MISMATCH`.
Restore the file and reinstall to fix.

Upgrade: if a prior AeonSetup.exe is available, install it first,
verify it works, then install the current AeonSetup.exe over it.
Verify: application binaries update, release identity updates, no
old runtime process remains, no duplicate release bundle remains.

Uninstall: from Add/Remove Programs, uninstall Aeon. Verify:
`C:\Program Files\Aeon\` (or per-user install root) is emptied of
Aeon files. `%LOCALAPPDATA%\Aeon` retention follows the declared
policy (for the research preview, the user is prompted; default is
to retain settings + not-yet-persisted transcripts). Reinstall must
succeed cleanly.

---

## WINDOWS-5 — Release-candidate closure

Set `docs/desktop/desktop_status.json.current_status =
FUNCTIONAL_RELEASE_CANDIDATE` only after ALL WINDOWS-0..4 steps have
executable evidence attached.

This report becomes the input to
`docs/desktop/DESKTOP_WINDOWS_ACCEPTANCE_REPORT.md`.
