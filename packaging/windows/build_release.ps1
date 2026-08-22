# build_release.ps1 — one-command Windows release for Aeon (WIN-PATCH-B).
#
# End-to-end flow performed by a Windows operator (no admin required):
#   1. Build dist\Aeon\Aeon.exe                         (build.ps1)
#   2. Generate RUNTIME_MANIFEST.json + .sha256 sidecar (build.ps1 already does this)
#   3. Bundle smoke test                                (verify_bundle.py)
#   4. Discover Inno Setup and build AeonSetup.exe     (build_installer.ps1)
#   5. Produce release ZIP containing only
#         AeonSetup.exe
#         AeonSetup.exe.sha256
#   6. Print absolute paths, byte sizes, and SHA-256s of every output.
#
# Fails closed at the first stage failure. Never silently reuses a stale
# installer, ZIP, manifest, or sidecar. File Explorer is optional and
# is only opened at the end when -OpenExplorer is passed.
#
# Usage:
#     powershell.exe -ExecutionPolicy Bypass -File packaging\windows\build_release.ps1
#     powershell.exe -ExecutionPolicy Bypass -File packaging\windows\build_release.ps1 -InnoCompiler "C:\Path\ISCC.exe" -OpenExplorer

param(
    [string]$InnoCompiler,
    [switch]$OpenExplorer
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root      = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$Pkg       = $PSScriptRoot
$DistRoot  = Join-Path $Root 'dist'
$AeonExe   = Join-Path $DistRoot 'Aeon\Aeon.exe'
$Manifest  = Join-Path $DistRoot 'Aeon\_internal\packaging\windows\RUNTIME_MANIFEST.json'
$Sidecar   = Join-Path $DistRoot 'Aeon\_internal\packaging\windows\RUNTIME_MANIFEST.sha256'
$InstallerOut = Join-Path $DistRoot 'installer'
$AeonSetup = Join-Path $InstallerOut 'AeonSetup.exe'
$AeonSetupSha = Join-Path $InstallerOut 'AeonSetup.exe.sha256'
$ReleaseZip = Join-Path $DistRoot 'Aeon_Desktop_7M_Research_Preview_Windows.zip'
$ReleaseZipSha = "$ReleaseZip.sha256"

function Write-Stage($label) {
    Write-Host ""
    Write-Host "==================================================================" -ForegroundColor Cyan
    Write-Host "[release] $label"                                                 -ForegroundColor Cyan
    Write-Host "==================================================================" -ForegroundColor Cyan
}

function Assert-Fresh([string]$path) {
    if (Test-Path $path) {
        Write-Host "[release] removing stale $path"
        Remove-Item -Force -Recurse $path
    }
}

function Show-Artifact([string]$path) {
    if (-not (Test-Path $path)) {
        throw "[release] expected artifact missing: $path"
    }
    $item = Get-Item $path
    $sha  = (Get-FileHash -Algorithm SHA256 $path).Hash.ToLowerInvariant()
    Write-Host ""
    Write-Host ("[release] artifact       : {0}" -f $item.FullName)
    Write-Host ("[release] size (bytes)   : {0}" -f $item.Length)
    Write-Host ("[release] sha256         : {0}" -f $sha)
    return @{ Path = $item.FullName; Bytes = $item.Length; Sha256 = $sha }
}

# ---------------------------------------------------------------------------
# Never silently reuse stale outputs.
# ---------------------------------------------------------------------------
Write-Stage 'clearing stale outputs (installer + release ZIP)'
Assert-Fresh $AeonSetup
Assert-Fresh $AeonSetupSha
Assert-Fresh $ReleaseZip
Assert-Fresh $ReleaseZipSha

# ---------------------------------------------------------------------------
# 1 + 2 + 3. build.ps1 runs the bundle build, the manifest generator,
# and the bundle smoke test. On failure it throws.
# ---------------------------------------------------------------------------
Write-Stage '1/5  building bundle (build.ps1)'
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Pkg 'build.ps1')
if ($LASTEXITCODE -ne 0) { throw '[release] build.ps1 failed' }
if (-not (Test-Path $AeonExe))   { throw "[release] Aeon.exe missing after build: $AeonExe" }
if (-not (Test-Path $Manifest))  { throw "[release] runtime manifest missing after build: $Manifest" }
if (-not (Test-Path $Sidecar))   { throw "[release] runtime manifest sidecar missing after build: $Sidecar" }

Write-Stage '2/5  verifying bundle (verify_bundle.py, second explicit pass)'
$Venv = Join-Path $Root '.build-venv'
$PyExe = Join-Path $Venv 'Scripts\python.exe'
if (-not (Test-Path $PyExe)) { throw "[release] build venv Python missing: $PyExe (run build.ps1 first)" }
& $PyExe (Join-Path $Pkg 'verify_bundle.py') --bundle (Join-Path $DistRoot 'Aeon')
if ($LASTEXITCODE -ne 0) { throw '[release] verify_bundle.py failed on the built bundle' }

# ---------------------------------------------------------------------------
# 4. Discover Inno Setup and build the installer.
# ---------------------------------------------------------------------------
Write-Stage '3/5  building installer (build_installer.ps1)'
if ($InnoCompiler) {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Pkg 'build_installer.ps1') -InnoCompiler $InnoCompiler
} else {
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $Pkg 'build_installer.ps1')
}
if ($LASTEXITCODE -ne 0) { throw '[release] build_installer.ps1 failed' }
$installerArt = Show-Artifact $AeonSetup
$installerShaArt = Show-Artifact $AeonSetupSha

# ---------------------------------------------------------------------------
# 5. Produce the release ZIP with ONLY AeonSetup.exe + its .sha256 sidecar.
# ---------------------------------------------------------------------------
Write-Stage '4/5  packaging release ZIP'
# Compress-Archive expects an array of source files, not a directory here —
# we deliberately do NOT include Aeon.exe / dist\Aeon\ etc. Only the
# installer + its digest ship.
Compress-Archive -Path $AeonSetup, $AeonSetupSha -DestinationPath $ReleaseZip -Force
if (-not (Test-Path $ReleaseZip)) { throw "[release] release ZIP not produced: $ReleaseZip" }

# Verify the ZIP contains ONLY those two entries.
Add-Type -AssemblyName System.IO.Compression.FileSystem | Out-Null
$zip = [System.IO.Compression.ZipFile]::OpenRead($ReleaseZip)
try {
    $entries = @($zip.Entries | ForEach-Object { $_.FullName })
} finally {
    $zip.Dispose()
}
$expected = @('AeonSetup.exe', 'AeonSetup.exe.sha256')
$extras = @($entries | Where-Object { $expected -notcontains $_ })
if ($extras.Count -gt 0) {
    throw "[release] release ZIP contains unexpected entries: $($extras -join ', ')"
}
foreach ($e in $expected) {
    if ($entries -notcontains $e) {
        throw "[release] release ZIP missing expected entry: $e"
    }
}
Set-Content -Path $ReleaseZipSha `
    -Value ("{0}  Aeon_Desktop_7M_Research_Preview_Windows.zip" -f `
        (Get-FileHash -Algorithm SHA256 $ReleaseZip).Hash.ToLowerInvariant()) `
    -Encoding ascii

$zipArt = Show-Artifact $ReleaseZip
$zipShaArt = Show-Artifact $ReleaseZipSha

# ---------------------------------------------------------------------------
# 5. Summary.
# ---------------------------------------------------------------------------
Write-Stage '5/5  DONE'
Write-Host ""
Write-Host "Outputs:"
Write-Host ("  installer : {0}" -f $installerArt.Path)
Write-Host ("              size={0} bytes  sha256={1}" -f $installerArt.Bytes, $installerArt.Sha256)
Write-Host ("  digest    : {0}" -f $installerShaArt.Path)
Write-Host ("  release   : {0}" -f $zipArt.Path)
Write-Host ("              size={0} bytes  sha256={1}" -f $zipArt.Bytes, $zipArt.Sha256)
Write-Host ("  digest    : {0}" -f $zipShaArt.Path)
Write-Host ""
Write-Host "ZIP contents (verified above):"
foreach ($e in $entries) { Write-Host "  - $e" }
Write-Host ""
Write-Host "This does NOT establish English quality — see packaging/windows/test_release.ps1 for launch verification." -ForegroundColor Yellow

if ($OpenExplorer) {
    # File Explorer is a convenience, not evidence. Only opened when
    # explicitly requested via -OpenExplorer.
    Start-Process explorer.exe $DistRoot | Out-Null
}
