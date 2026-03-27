<#
.SYNOPSIS
    Secure Federated Learning - Windows Launcher
.EXAMPLE
    .\run.ps1
    .\run.ps1 -NumClients 10 -NumRounds 7 -NoDropout
    .\run.ps1 -NumClients 10 -NumRounds 5 -NoDropout -AttackType label_flip -AttackFrac 0.2
    .\run.ps1 -NumClients 10 -NumRounds 5 -NonIID
#>
param(
    [int]   $NumClients = 10,
    [int]   $NumRounds  = 5,
    [switch]$NoDropout,
    [string]$AttackType = "none",
    [float] $AttackFrac = 0.0,
    [switch]$NonIID
)

$ErrorActionPreference = "Stop"

$NumMalicious    = [math]::Floor($NumClients * $AttackFrac)
$NumHonest       = $NumClients - $NumMalicious
$GanacheAccounts = $NumClients + 5
$LogDir          = "logs"
$ResultsDir      = "results"
$ScriptDir       = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe       = "C:\Users\kapil\miniconda3\envs\myenv\python.exe"
$GancheScript    = (Get-Command ganache -ErrorAction Stop).Source
$TruffleScript   = (Get-Command truffle -ErrorAction Stop).Source

New-Item -ItemType Directory -Force -Path $LogDir     | Out-Null
New-Item -ItemType Directory -Force -Path $ResultsDir | Out-Null

function Test-Port([int]$port) {
    try { $t = New-Object System.Net.Sockets.TcpClient; $t.Connect("127.0.0.1",$port); $t.Close(); return $true }
    catch { return $false }
}
function Stop-ByPattern([string]$pat) {
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like "*$pat*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}
function Stop-OnPort([int]$port) {
    Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}

$dropoutLabel = if ($NoDropout)             { 'disabled'           } else { 'enabled' }
$attackLabel  = if ($AttackType -eq 'none') { 'none'               } else { '{0} - {1} clients' -f $AttackType, $NumMalicious }
$dataLabel    = if ($NonIID)                { 'non-IID (Dirichlet)' } else { 'IID' }

Write-Host ""
Write-Host "============================================================"
Write-Host "  Secure Federated Learning - Windows Launcher"
Write-Host "============================================================"
Write-Host ('  Clients : {0}  ({1} honest, {2} malicious)' -f $NumClients, $NumHonest, $NumMalicious)
Write-Host "  Rounds  : $NumRounds"
Write-Host "  Dropout : $dropoutLabel"
Write-Host "  Attacks : $attackLabel"
Write-Host "  Data    : $dataLabel"
Write-Host "============================================================"
Write-Host ""

# ── Step 1: Kill old processes ────────────────────────────────
Write-Host "[1/5] Cleaning up old processes..."
Stop-ByPattern "client.py"
Stop-ByPattern "server.py"
Stop-ByPattern "attack_simulator.py"
Stop-ByPattern "ganache"
Stop-OnPort 7545
Stop-OnPort 9090
Start-Sleep -Seconds 2
Write-Host "  Done."

# ── Step 2: Start Ganache (via powershell -File since it's a .ps1) ──
Write-Host ""
Write-Host ('  [2/5] Starting Ganache (port 7545, {0} accounts)...' -f $GanacheAccounts)
$ganacheLog = Join-Path $ScriptDir "$LogDir\ganache.log"
$ganacheProc = Start-Process powershell.exe `
    -ArgumentList "-ExecutionPolicy","Bypass","-File","`"$GancheScript`"","--port","7545","--accounts",$GanacheAccounts `
    -RedirectStandardOutput $ganacheLog `
    -RedirectStandardError  (Join-Path $ScriptDir "$LogDir\ganache_err.log") `
    -NoNewWindow -PassThru

$ready = $false
for ($i = 1; $i -le 15; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Port 7545) { $ready = $true; break }
    if ($ganacheProc.HasExited) { break }
}
if (-not $ready) {
    Write-Host "  ERROR: Ganache failed. Check $LogDir\ganache.log"
    exit 1
}
Write-Host ('  Ganache ready (PID: {0})' -f $ganacheProc.Id)

# ── Step 3: Deploy contracts (truffle is also a .ps1) ────────
Write-Host ""
Write-Host "  [3/5] Deploying smart contracts..."
$truffleLog = Join-Path $ScriptDir "$LogDir\truffle.log"
$truffleProc = Start-Process powershell.exe `
    -ArgumentList "-ExecutionPolicy","Bypass","-File","`"$TruffleScript`"","migrate","--reset","--network","development" `
    -RedirectStandardOutput $truffleLog `
    -RedirectStandardError  (Join-Path $ScriptDir "$LogDir\truffle_err.log") `
    -NoNewWindow -PassThru -Wait
if ($truffleProc.ExitCode -ne 0) {
    Write-Host "  ERROR: Truffle failed. Check $LogDir\truffle.log"
    Stop-Process -Id $ganacheProc.Id -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host "  Contracts deployed."

# ── Step 4: Start FL server ───────────────────────────────────
Write-Host ""
Write-Host ('[4/5] Starting FL server ({0} clients, {1} rounds)...' -f $NumClients, $NumRounds)
$serverLog = Join-Path $ScriptDir "$LogDir\server.log"
$serverProc = Start-Process $PythonExe `
    -ArgumentList "server.py","$NumClients","$NumRounds" `
    -WorkingDirectory $ScriptDir `
    -RedirectStandardOutput $serverLog `
    -RedirectStandardError  (Join-Path $ScriptDir "$LogDir\server_err.log") `
    -NoNewWindow -PassThru

$ready = $false
for ($i = 1; $i -le 20; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Port 9090) { $ready = $true; break }
    if ($serverProc.HasExited) { break }
}
if (-not $ready) {
    Write-Host "  ERROR: Server failed. Check $LogDir\server.log"
    Stop-Process -Id $ganacheProc.Id -Force -ErrorAction SilentlyContinue
    exit 1
}
Write-Host ('  Server ready (PID: {0})' -f $serverProc.Id)

# ── Step 5: Launch clients ────────────────────────────────────
Write-Host ""
Write-Host ('[5/5] Launching {0} clients...' -f $NumClients)
$clientProcs = @()

for ($i = 0; $i -lt $NumHonest; $i++) {
    $pArgs = @("client.py","$i","$NumClients")
    if ($NoDropout)  { $pArgs += "--no-dropout" }
    if ($NonIID)     { $pArgs += "--non-iid" }
    $p = Start-Process $PythonExe -ArgumentList $pArgs `
        -WorkingDirectory $ScriptDir `
        -RedirectStandardOutput (Join-Path $ScriptDir "$LogDir\client_$i.log") `
        -RedirectStandardError  (Join-Path $ScriptDir "$LogDir\client_${i}_err.log") `
        -NoNewWindow -PassThru
    $clientProcs += $p
    Write-Host ('  Honest  client {0} started (PID: {1})' -f $i, $p.Id)
}

if ($NumMalicious -gt 0 -and $AttackType -ne "none") {
    for ($i = $NumHonest; $i -lt $NumClients; $i++) {
        $p = Start-Process $PythonExe `
            -ArgumentList "attack_simulator.py","$i","$AttackType","$NumClients" `
            -WorkingDirectory $ScriptDir `
            -RedirectStandardOutput (Join-Path $ScriptDir "$LogDir\client_$i.log") `
            -RedirectStandardError  (Join-Path $ScriptDir "$LogDir\client_${i}_err.log") `
            -NoNewWindow -PassThru
        $clientProcs += $p
        Write-Host ('  ATTACK  client {0} ({1}) started (PID: {2})' -f $i, $AttackType, $p.Id)
    }
}

Write-Host ""
Write-Host "============================================================"
Write-Host "  All launched! Streaming server log... (Ctrl+C to stop watching)"
Write-Host "============================================================"
Write-Host ""

# ── Stream server log until server exits ─────────────────────
$lastLine = 0
while (-not $serverProc.HasExited) {
    $lines = Get-Content $serverLog -ErrorAction SilentlyContinue
    if ($lines -and $lines.Count -gt $lastLine) {
        $lines[$lastLine..($lines.Count - 1)] | ForEach-Object { Write-Host $_ }
        $lastLine = $lines.Count
    }
    Start-Sleep -Seconds 1
}
$lines = Get-Content $serverLog -ErrorAction SilentlyContinue
if ($lines -and $lines.Count -gt $lastLine) {
    $lines[$lastLine..($lines.Count - 1)] | ForEach-Object { Write-Host $_ }
}

# ── Cleanup ───────────────────────────────────────────────────
Write-Host ""
Write-Host "  Training complete. Cleaning up..."
$clientProcs | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
Stop-Process -Id $ganacheProc.Id -Force -ErrorAction SilentlyContinue
Stop-ByPattern "client.py"
Stop-ByPattern "attack_simulator.py"
Write-Host "  Results saved to: $ResultsDir\"
Write-Host ""
