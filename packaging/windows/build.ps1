# build.ps1 — reproducible PyInstaller build for Aeon on Windows x64.
#
# Run from an elevated-not-required PowerShell:
#     powershell.exe -ExecutionPolicy Bypass -File packaging\windows\build.ps1
#
# Prerequisites (§14 automation):
#   * Windows 10 or Windows 11 x64
#   * Python 3.11 x64 (matches the pin in pyproject.toml)
#   * Access to https://download.pytorch.org/whl/cpu (for torch CPU wheel)
#   * pip / venv
#   * Inno Setup 6.x installed (for W6 installer build; used by
#     build_installer.ps1, not by this script)
#
# This script never installs Aeon runtime-time deps from within the frozen
# app. It only prepares the BUILD environment.

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root      = Resolve-Path (Join-Path $PSScriptRoot '..\..' )
$Pkg       = Join-Path $Root 'packaging\windows'
$DistRoot  = Join-Path $Root 'dist'
$BuildRoot = Join-Path $Root 'build'
$Venv      = Join-Path $Root '.build-venv'
$Requirements = Join-Path $Pkg 'requirements-windows.lock'

Write-Host "[build] Aeon Windows CPU build" -ForegroundColor Cyan
Write-Host "[build] repo root       = $Root"
Write-Host "[build] output          = $DistRoot"

# 1. Create an isolated build venv
if (-not (Test-Path $Venv)) {
    Write-Host "[build] creating venv at $Venv"
    py -3.11 -m venv $Venv
}
& (Join-Path $Venv 'Scripts\python.exe') -m pip install --upgrade pip

# 2. Install pinned deps. torch==2.5.1+cpu is only on the pytorch CPU index,
#    not PyPI — must pass --extra-index-url or the lock resolves to nothing.
if (Test-Path $Requirements) {
    & (Join-Path $Venv 'Scripts\python.exe') -m pip install `
        --extra-index-url https://download.pytorch.org/whl/cpu `
        -r $Requirements
    if ($LASTEXITCODE -ne 0) { throw "pip install failed against $Requirements" }
} else {
    Write-Warning "requirements-windows.lock missing — installing minimal pins"
    & (Join-Path $Venv 'Scripts\python.exe') -m pip install `
        torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
    & (Join-Path $Venv 'Scripts\python.exe') -m pip install `
        safetensors sentencepiece pyyaml "numpy<2" pyinstaller==6.11.1
}

# 3. Run the full regression suite (must be 155/155 or newer) BEFORE building.
Write-Host "[build] running regression suite" -ForegroundColor Cyan
$env:PYTHONPATH = $Root
$suite = @(
    'test_substrate_port','test_aeon_sanity','test_tokenizer',
    'test_feedback','test_feedback_diagnostics','test_six_patches',
    'test_recursion_topology','test_stream_independence',
    'test_config_invariants','test_observability','test_checkpoint',
    'test_diagnose','test_threat_model','test_provenance',
    'test_protected_checkpoint','test_runtime_policy','test_continuity',
    'test_adversarial','test_evidence_hygiene',
    'test_entry','test_launcher_and_job'
)
$total = 0
foreach ($t in $suite) {
    $out = & (Join-Path $Venv 'Scripts\python.exe') (Join-Path $Root "tests\$t.py")
    $n = ($out | Select-String -Pattern '^\d+ checks passed' | Select-Object -First 1) -replace ' checks passed.*',''
    if ($n) { $total += [int]$n }
    Write-Host "  $t : $n"
}
Write-Host "[build] test total: $total"

# 4. W10-10/A23: real third-party licences are REQUIRED. The old build
#    silently created a PLACEHOLDER.txt "Place third-party licences here
#    before shipping." and continued — that meant an installer could ship
#    with no upstream notices. We now:
#      * refuse the build if packaging\windows\licenses\ does not exist,
#      * refuse if the directory contains no files or only PLACEHOLDER.txt,
#      * refuse if any required license file is missing (torch, pyinstaller,
#        sentencepiece, safetensors, numpy, pyyaml — the runtime deps).
#    Operators must populate the licenses/ directory from the actual
#    upstream sources before invoking build.ps1.
$Licenses = Join-Path $Pkg 'licenses'
if (-not (Test-Path $Licenses)) {
    throw "[build] licences directory missing at $Licenses. W10-10/A23: populate it with the real third-party notices for torch, sentencepiece, safetensors, numpy, pyyaml, and pyinstaller before running build.ps1."
}
$LicenseFiles = Get-ChildItem -Path $Licenses -File | Where-Object { $_.Name -ne 'PLACEHOLDER.txt' }
if (-not $LicenseFiles -or $LicenseFiles.Count -eq 0) {
    throw "[build] licences directory at $Licenses is empty (or only contains PLACEHOLDER.txt). W10-10/A23: real third-party licences required — refusing to ship without them."
}
$RequiredLicenses = @('torch', 'pyinstaller', 'sentencepiece', 'safetensors', 'numpy', 'pyyaml')
foreach ($req in $RequiredLicenses) {
    $match = $LicenseFiles | Where-Object { $_.Name -match "(?i)$req" }
    if (-not $match) {
        throw "[build] licences directory at $Licenses is missing the license for '$req'. W10-10/A23: every runtime dependency must have its upstream LICENSE file present before the installer can be built."
    }
}
Write-Host "[build] licences OK — $($LicenseFiles.Count) files, all required deps covered"

# 5. Prepare release metadata
& (Join-Path $Venv 'Scripts\python.exe') (Join-Path $Pkg 'release_metadata.py') `
    --out (Join-Path $Pkg 'RELEASE.json') --build-type release

# 6. PyInstaller build
Write-Host "[build] pyinstaller" -ForegroundColor Cyan
if (Test-Path $BuildRoot) { Remove-Item -Recurse -Force $BuildRoot }
if (Test-Path (Join-Path $DistRoot 'Aeon')) { Remove-Item -Recurse -Force (Join-Path $DistRoot 'Aeon') }
& (Join-Path $Venv 'Scripts\pyinstaller.exe') --clean (Join-Path $Pkg 'Aeon.spec')
if ($LASTEXITCODE -ne 0) { throw "pyinstaller failed" }

# 7. Manifest generation MUST happen before the bundle-smoke test, because
#    Aeon.exe --verify-installation reads the manifest.
Write-Host "[build] runtime manifest" -ForegroundColor Cyan
& (Join-Path $Venv 'Scripts\python.exe') (Join-Path $Pkg 'generate_runtime_manifest.py') `
    --bundle (Join-Path $DistRoot 'Aeon') `
    --release (Join-Path $Pkg 'RELEASE.json')
if ($LASTEXITCODE -ne 0) { throw "generate_runtime_manifest failed" }

# 8. Bundle-smoke test — Aeon.exe --version, --verify-installation, --worker.
#    A non-zero exit here stops the build; the workflow's A6 step runs it too.
Write-Host "[build] bundle smoke test" -ForegroundColor Cyan
& (Join-Path $Venv 'Scripts\python.exe') (Join-Path $Pkg 'verify_bundle.py') `
    --bundle (Join-Path $DistRoot 'Aeon')
if ($LASTEXITCODE -ne 0) { throw "verify_bundle failed" }

Write-Host "[build] DONE — dist\Aeon\Aeon.exe" -ForegroundColor Green
