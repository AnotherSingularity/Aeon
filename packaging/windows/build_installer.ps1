# build_installer.ps1 — build AeonSetup.exe via Inno Setup (§W6).
#
# Prerequisites:
#   * Inno Setup 6.x installed. Search order (WIN-PATCH-A/Failure C):
#       1. -InnoCompiler <path>  (explicit parameter override)
#       2. $env:AEON_ISCC        (environment override)
#       3. Get-Command ISCC.exe  (compiler on PATH)
#       4. %ProgramFiles(x86)%\Inno Setup 6\ISCC.exe (system-wide, x86)
#       5. %ProgramFiles%\Inno Setup 6\ISCC.exe     (system-wide, x64)
#       6. %LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe  (per-user)
#   * packaging\windows\build.ps1 already produced dist\Aeon\Aeon.exe and
#     dist\Aeon\_internal\packaging\windows\RUNTIME_MANIFEST.json (PyInstaller 6.x onedir)
#
# Output: dist\installer\AeonSetup.exe
#         dist\installer\AeonSetup.exe.sha256

param(
    [string]$InnoCompiler
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
$Iss = Join-Path $PSScriptRoot 'AeonInstaller.iss'
$Bundle = Join-Path $Root 'dist\Aeon\Aeon.exe'
$Manifest = Join-Path $Root 'dist\Aeon\_internal\packaging\windows\RUNTIME_MANIFEST.json'
$Sidecar = Join-Path $Root 'dist\Aeon\_internal\packaging\windows\RUNTIME_MANIFEST.sha256'
$InstallerOut = Join-Path $Root 'dist\installer'

if (-not (Test-Path $Bundle)) {
    throw "PyInstaller bundle missing: $Bundle. Run build.ps1 first."
}
if (-not (Test-Path $Manifest)) {
    throw "Runtime manifest missing: $Manifest. Run build.ps1 first."
}
if (-not (Test-Path $Sidecar)) {
    throw "Runtime manifest sidecar missing: $Sidecar. Run build.ps1 first (generate_runtime_manifest.py emits it)."
}

# WIN-PATCH-A/Failure C: deterministic Inno Setup discovery.
function Resolve-Iscc {
    param([string]$Explicit)

    $checked = New-Object System.Collections.Generic.List[string]

    if ($Explicit) {
        $checked.Add("(param -InnoCompiler) $Explicit") | Out-Null
        if (Test-Path $Explicit) { return @{ Path = $Explicit; Source = 'param -InnoCompiler' } }
    }

    $envIscc = $env:AEON_ISCC
    if ($envIscc) {
        $checked.Add("(`$env:AEON_ISCC) $envIscc") | Out-Null
        if (Test-Path $envIscc) { return @{ Path = $envIscc; Source = 'env AEON_ISCC' } }
    }

    $onPath = $null
    try { $onPath = (Get-Command ISCC.exe -ErrorAction Stop).Source } catch { $onPath = $null }
    if ($onPath) {
        $checked.Add("(Get-Command ISCC.exe) $onPath") | Out-Null
        return @{ Path = $onPath; Source = 'Get-Command ISCC.exe' }
    } else {
        $checked.Add('(Get-Command ISCC.exe) not on PATH') | Out-Null
    }

    $candidates = @()
    if (${env:ProgramFiles(x86)}) {
        $candidates += @{ Src = 'ProgramFiles(x86)'; P = (Join-Path ${env:ProgramFiles(x86)} 'Inno Setup 6\ISCC.exe') }
    }
    if ($env:ProgramFiles) {
        $candidates += @{ Src = 'ProgramFiles';       P = (Join-Path $env:ProgramFiles 'Inno Setup 6\ISCC.exe') }
    }
    if ($env:LOCALAPPDATA) {
        $candidates += @{ Src = 'LOCALAPPDATA';       P = (Join-Path $env:LOCALAPPDATA 'Programs\Inno Setup 6\ISCC.exe') }
    }
    foreach ($c in $candidates) {
        $checked.Add("($($c.Src)) $($c.P)") | Out-Null
        if (Test-Path $c.P) { return @{ Path = $c.P; Source = $c.Src } }
    }

    $msg = "[installer] Inno Setup 6 not found. Checked (in order):`n  - " + ($checked -join "`n  - ") + `
        "`nInstall Inno Setup 6.x from https://jrsoftware.org/isinfo.php " + `
        "(a per-user install into %LOCALAPPDATA%\Programs\Inno Setup 6 works). " + `
        "Do NOT auto-download. The script will not install Inno Setup for you."
    throw $msg
}

$Resolved = Resolve-Iscc -Explicit $InnoCompiler
$Iscc = $Resolved.Path
Write-Host "[installer] ISCC path    = $Iscc" -ForegroundColor Cyan
Write-Host "[installer] discovered via $($Resolved.Source)"

# Print compiler version for the release log. ISCC.exe /? prints its
# banner (including version) to stdout even when no arguments are
# given; capture and echo the first line so operators can confirm.
try {
    $isccInfo = & $Iscc /? 2>&1 | Select-Object -First 3
    foreach ($line in $isccInfo) { Write-Host "[installer] $line" }
} catch {
    Write-Warning "[installer] could not query ISCC version banner: $($_.Exception.Message)"
}

if (-not (Test-Path $InstallerOut)) {
    New-Item -ItemType Directory -Path $InstallerOut | Out-Null
} else {
    # WIN-PATCH-A: never silently reuse a stale installer.
    $stale = Join-Path $InstallerOut 'AeonSetup.exe'
    if (Test-Path $stale) { Remove-Item -Force $stale }
    $staleSidecar = Join-Path $InstallerOut 'AeonSetup.exe.sha256'
    if (Test-Path $staleSidecar) { Remove-Item -Force $staleSidecar }
}

Write-Host "[installer] ISCC $Iss" -ForegroundColor Cyan
& $Iscc $Iss
if ($LASTEXITCODE -ne 0) { throw 'ISCC failed' }

$AeonSetup = Join-Path $InstallerOut 'AeonSetup.exe'
if (Test-Path $AeonSetup) {
    $sz = (Get-Item $AeonSetup).Length
    $sha = (Get-FileHash -Algorithm SHA256 $AeonSetup).Hash.ToLowerInvariant()
    Write-Host "[installer] AeonSetup.exe   size=$sz bytes"
    Write-Host "[installer] sha256          $sha"
    Set-Content -Path (Join-Path $InstallerOut 'AeonSetup.exe.sha256') `
        -Value "$sha  AeonSetup.exe" -Encoding ascii
} else {
    throw "AeonSetup.exe not produced"
}
