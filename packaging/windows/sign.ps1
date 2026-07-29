# sign.ps1 — code-signing pipeline for Aeon.exe and AeonSetup.exe (§W7).
#
# Signing credentials come from EXTERNAL secure build secrets. This script
# NEVER commits keys, certificates, passwords, or displays them. The
# expected environment variables (set by the build host, NOT the repo):
#
#     AEON_SIGNTOOL_PATH   -> full path to signtool.exe (Windows SDK)
#     AEON_SIGN_CERT_PFX   -> path to the signing PFX (held out-of-repo)
#     AEON_SIGN_CERT_PASS  -> read from a secret store; NEVER logged
#     AEON_SIGN_TIMESTAMP  -> RFC 3161 timestamp URL, e.g. http://timestamp.digicert.com
#
# Hardware-backed signing (recommended for release): pass the vendor's
# CSP / KSP arguments to signtool. Do NOT bake the CSP name into this script;
# it is deployment-specific.

param(
    [string]$AeonExe = "dist\Aeon\Aeon.exe",
    [string]$Installer = "dist\installer\AeonSetup.exe"
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$signtool = $env:AEON_SIGNTOOL_PATH
if (-not $signtool -or -not (Test-Path $signtool)) {
    throw 'AEON_SIGNTOOL_PATH not set or invalid. Point it at signtool.exe from the Windows SDK.'
}
$pfx = $env:AEON_SIGN_CERT_PFX
if (-not $pfx) {
    throw 'AEON_SIGN_CERT_PFX not set. Signing REQUIRES an out-of-repo PFX.'
}
if (-not (Test-Path $pfx)) {
    throw "AEON_SIGN_CERT_PFX file missing: $pfx"
}
$pass = $env:AEON_SIGN_CERT_PASS
if (-not $pass) {
    throw 'AEON_SIGN_CERT_PASS not set. Retrieve it from your secret store (never commit).'
}
$ts = $env:AEON_SIGN_TIMESTAMP
if (-not $ts) { $ts = 'http://timestamp.digicert.com' }

function Invoke-SignOne([string]$path) {
    if (-not (Test-Path $path)) { throw "cannot sign — file missing: $path" }
    Write-Host "[sign] $path" -ForegroundColor Cyan
    # /fd sha256   file-digest
    # /td sha256   timestamp-digest
    # /tr URL      RFC 3161 timestamp URL
    # /f PFX       PFX cert file
    # /p PASSWORD  PFX password (from env — never logged; PS does not echo)
    & $signtool sign /fd sha256 /td sha256 /tr $ts /f $pfx /p $pass $path | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "signtool failed on $path" }
    & $signtool verify /pa /v $path | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "signtool verify failed on $path" }
    Write-Host "[sign] OK $path"
}

Invoke-SignOne $AeonExe
Invoke-SignOne $Installer
Write-Host "[sign] all payloads signed" -ForegroundColor Green
