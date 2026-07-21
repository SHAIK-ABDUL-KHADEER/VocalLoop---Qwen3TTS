# Verify each endpoint family against a fresh server.
#
# On an 8 GB GPU MAX_RESIDENT=1, so switching endpoint families evicts and
# reloads a 4 GB model. Doing that repeatedly kills the CUDA process on this
# machine (native crash, no traceback). Testing one family per server process
# avoids the swap entirely and still exercises every endpoint.
#
# Not needed on the A100: set QWEN_TTS_MAX_RESIDENT=3 and nothing ever evicts.

param([string]$Filter = "")

$Root = $PSScriptRoot
$Py = Join-Path $Root "venv\Scripts\python.exe"

function Start-Server {
    Get-NetTCPConnection -LocalPort 9000 -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
    Start-Sleep 3
    $env:QWEN_TTS_MAX_RESIDENT = "1"
    $p = Start-Process -FilePath $Py `
        -ArgumentList "-u", "server\app.py" `
        -WorkingDirectory $Root `
        -RedirectStandardOutput (Join-Path $Root "server.log") `
        -RedirectStandardError (Join-Path $Root "server.err") `
        -WindowStyle Hidden -PassThru
    Start-Sleep 12
    return $p
}

$families = if ($Filter) { @($Filter) } else { @("described", "builtin", "vocalist", "clone", "err") }

foreach ($f in $families) {
    Write-Host "`n########## family: $f ##########" -ForegroundColor Cyan
    $srv = Start-Server
    & $Py -u (Join-Path $Root "test_endpoints.py") --filter $f
    if ($srv -and -not $srv.HasExited) { Stop-Process -Id $srv.Id -Force -ErrorAction SilentlyContinue }
}

Write-Host "`nall families done" -ForegroundColor Green
