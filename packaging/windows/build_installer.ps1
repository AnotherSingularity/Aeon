# build_installer.ps1 — build AeonSetup.exe via Inno Setup (§W6).
#
# Prerequisites:
#   * Inno Setup 6.x installed (default path: %ProgramFiles(x86)%\Inno Setup 6)
#   * packaging\windows\build.ps1 already produced dist\Aeon\Aeon.exe and
#     dist\Aeon\packaging\windows\RUNTIME_MANIFEST.json
#
# Output: dist\installer\AeonSetup.exe

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$Iss = Join-Path $PSScriptRoot 'AeonInstaller.iss'
$Bundle = Join-Path $Root 'dist\Aeon\Aeon.exe'
$Manifest = Join-Path $Root 'dist\Aeon\packaging\windows\RUNTIME_MANIFEST.json'
$InstallerOut = Join-Path $Root 'dist\installer'

if (-not (Test-Path $Bundle)) {
    throw "PyInstaller bundle missing: $Bundle. Run build.ps1 first."
}
if (-not (Test-Path $Manifest)) {
    throw "Runtime manifest missing: $Manifest. Run build.ps1 first."
}

$Iscc = Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe'
if (-not (Test-Path $Iscc)) {
    $Iscc = Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe'
}
if (-not (Test-Path $Iscc)) {
    throw 'ISCC.exe not found. Install Inno Setup 6.x from https://jrsoftware.org/isinfo.php'
}

if (-not (Test-Path $InstallerOut)) {
    New-Item -ItemType Directory -Path $InstallerOut | Out-Null
}

Write-Host "[installer] ISCC $Iss" -ForegroundColor Cyan
& $Iscc $Iss
if ($LASTEXITCODE -ne 0) { throw 'ISCC failed' }

$AeonSetup = Join-Path $InstallerOut 'AeonSetup.exe'
if (Test-Path $AeonSetup) {
    $sz = (Get-Item $AeonSetup).Length
    $sha = (Get-FileHash -Algorithm SHA256 $AeonSetup).Hash
    Write-Host "[installer] AeonSetup.exe   size=$sz bytes"
    Write-Host "[installer] sha256          $sha"
    Set-Content -Path (Join-Path $InstallerOut 'AeonSetup.exe.sha256') `
        -Value "$sha  AeonSetup.exe" -Encoding ascii
} else {
    throw "AeonSetup.exe not produced"
}
