# test_release.ps1 — Windows launch verification for a completed bundle
# (WIN-PATCH-B).
#
# Verifies a built dist\Aeon bundle by executing:
#     .\dist\Aeon\Aeon.exe --version
#     .\dist\Aeon\Aeon.exe --verify-installation
#
# With -LaunchChat, also starts Aeon.exe --chat, confirms the process
# remains alive for a short window, reports the process ID and exe
# path, and terminates cleanly. This proves STARTUP only. It does NOT
# claim conversational quality has been verified.
#
# Usage:
#     powershell.exe -ExecutionPolicy Bypass -File packaging\windows\test_release.ps1
#     powershell.exe -ExecutionPolicy Bypass -File packaging\windows\test_release.ps1 -LaunchChat
#     powershell.exe -ExecutionPolicy Bypass -File packaging\windows\test_release.ps1 -BundleRoot 'C:\path\to\Aeon' -LaunchChat -ChatDwellSeconds 4

param(
    [string]$BundleRoot,
    [switch]$LaunchChat,
    [int]$ChatDwellSeconds = 3
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

if (-not $BundleRoot) {
    $Root = Resolve-Path (Join-Path $PSScriptRoot '..\..')
    $BundleRoot = Join-Path $Root 'dist\Aeon'
}
$BundleRoot = (Resolve-Path $BundleRoot).Path
$Exe = Join-Path $BundleRoot 'Aeon.exe'
if (-not (Test-Path $Exe)) { throw "[test_release] Aeon.exe missing: $Exe" }

function Invoke-Aeon($argv) {
    Write-Host "[test_release] $Exe $($argv -join ' ')" -ForegroundColor Cyan
    $out = & $Exe @argv 2>&1
    $rc = $LASTEXITCODE
    if ($out) { $out | ForEach-Object { Write-Host "  $_" } }
    return $rc
}

# --- 1. --version -----------------------------------------------------------
$rc = Invoke-Aeon @('--version')
if ($rc -ne 0) { throw "[test_release] Aeon.exe --version failed (rc=$rc)" }

# --- 2. --verify-installation ----------------------------------------------
$rc = Invoke-Aeon @('--verify-installation')
if ($rc -ne 0) { throw "[test_release] Aeon.exe --verify-installation failed (rc=$rc)" }

# --- 3. Optional chat launch (opt-in via -LaunchChat) ----------------------
if ($LaunchChat) {
    Write-Host "[test_release] launching $Exe --chat (dwell $ChatDwellSeconds s)" -ForegroundColor Cyan
    # Snapshot Aeon.exe processes BEFORE launch so we can detect hidden
    # duplicates left behind by earlier runs and avoid attributing them
    # to this test.
    $before = @(Get-Process -Name 'Aeon' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
    $proc = Start-Process -FilePath $Exe -ArgumentList '--chat' -PassThru
    if (-not $proc) { throw '[test_release] Start-Process returned no process object' }
    try {
        Start-Sleep -Seconds $ChatDwellSeconds
        $proc.Refresh()
        if ($proc.HasExited) {
            throw "[test_release] Aeon --chat exited early (rc=$($proc.ExitCode))"
        }
        Write-Host ("[test_release] Aeon --chat alive:  pid={0}  exe={1}" -f $proc.Id, $Exe)
        Write-Host "[test_release] This proves startup only. Conversational quality is NOT verified by this test." -ForegroundColor Yellow
    } finally {
        # Terminate cleanly. Do not orphan the process.
        if (-not $proc.HasExited) {
            try { $proc.CloseMainWindow() | Out-Null } catch {}
            Start-Sleep -Milliseconds 500
            if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue }
        }
        # Kill any Aeon.exe left over that WAS spawned by this run (not
        # a pre-existing one). This prevents hidden duplicate instances
        # from lingering after the test.
        $after = @(Get-Process -Name 'Aeon' -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Id)
        foreach ($id in $after) {
            if ($before -notcontains $id -and $id -ne $proc.Id) {
                Write-Host "[test_release] terminating leftover Aeon.exe pid=$id"
                Stop-Process -Id $id -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

Write-Host ""
Write-Host "[test_release] OK — startup verified (version, verify-installation$( if ($LaunchChat) { ', chat' }))" -ForegroundColor Green
