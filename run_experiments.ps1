<#
.SYNOPSIS
    Batch Experiment Runner — Secure Federated Learning (Windows)
.DESCRIPTION
    Windows PowerShell equivalent of run_experiments.sh.
    Runs FL experiments from 10→20 clients for round configs 5, 7, 10.
    Total: 11 clients × 3 round configs = 33 experiments.
.PARAMETER NoDropout
    Switch — disable client dropout simulation
.PARAMETER AttackType
    Attack type: none | label_flip | scaling | noise_injection (default: none)
.PARAMETER AttackFrac
    Fraction of clients that are malicious, 0.0–0.5 (default: 0.0)
.EXAMPLE
    .\run_experiments.ps1
    .\run_experiments.ps1 -NoDropout
    .\run_experiments.ps1 -NoDropout -AttackType label_flip -AttackFrac 0.2
    .\run_experiments.ps1 -NoDropout -AttackType scaling -AttackFrac 0.3
#>

param(
    [switch]$NoDropout,
    [string]$AttackType = "none",
    [float] $AttackFrac = 0.0
)

$ErrorActionPreference = "Continue"

$RoundConfigs = @(5, 7, 10)
$MinClients   = 10
$MaxClients   = 20
$DropoutFlag  = if ($NoDropout) { "--no-dropout" } else { "" }

$Timestamp      = Get-Date -Format "yyyyMMdd_HHmmss"
$BatchSuffix    = if ($NoDropout) { "_nodrop" } else { "" }
if ($AttackType -ne "none") {
    $fracStr     = "$AttackFrac" -replace '\.', 'p'
    $BatchSuffix += "_${AttackType}_${fracStr}"
}
$ExperimentDir = "experiment_results"
$BatchDir      = "$ExperimentDir\batch_${Timestamp}${BatchSuffix}"
$TotalRuns     = ($MaxClients - $MinClients + 1) * $RoundConfigs.Count
$CurrentRun    = 0
$Completed     = 0
$Failed        = 0
$ScriptDir     = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonExe     = "C:\Users\kapil\miniconda3\envs\myenv\python.exe"
$GancheScript  = (Get-Command ganache -ErrorAction Stop).Source
$TruffleScript = (Get-Command truffle -ErrorAction Stop).Source

New-Item -ItemType Directory -Force -Path $BatchDir | Out-Null

# ── Helper: test if a TCP port is open ───────────────────────
function Test-Port([int]$port) {
    try {
        $tcp = New-Object System.Net.Sockets.TcpClient
        $tcp.Connect("127.0.0.1", $port)
        $tcp.Close()
        return $true
    } catch { return $false }
}

# ── Helper: kill processes matching a command-line pattern ───
function Stop-ByPattern([string]$pattern) {
    Get-CimInstance Win32_Process |
        Where-Object { $_.CommandLine -like "*$pattern*" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
}

# ── Helper: kill whatever holds a port ───────────────────────
function Stop-OnPort([int]$port) {
    Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
        ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
}

$BatchStart = Get-Date

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗"
Write-Host "║           BATCH EXPERIMENT RUNNER (Windows)             ║"
Write-Host "╠══════════════════════════════════════════════════════════╣"
Write-Host "║  Clients range : $MinClients → $MaxClients (incrementing by 1)"
Write-Host "║  Round configs : 5, 7, 10"
$dropoutLabel = if ($NoDropout)             { 'DISABLED'          } else { 'enabled' }
$attackLabel  = if ($AttackType -eq 'none') { 'none (all honest)'  } else { '{0} @ {1} fraction' -f $AttackType, $AttackFrac }
Write-Host "║  Dropout       : $dropoutLabel"
Write-Host "║  Attacks       : $attackLabel"
Write-Host "║  Total runs    : $TotalRuns"
Write-Host "║  Output dir    : $BatchDir"
Write-Host "╚══════════════════════════════════════════════════════════╝"
Write-Host ""

foreach ($NumRounds in $RoundConfigs) {
    Write-Host ""
    Write-Host "╔══════════════════════════════════════════════════════════╗"
    Write-Host "║  PHASE: $NumRounds ROUNDS — Clients ${MinClients}→${MaxClients}"
    Write-Host "╚══════════════════════════════════════════════════════════╝"

    for ($NumClients = $MinClients; $NumClients -le $MaxClients; $NumClients++) {
        $CurrentRun++
        $RunLabel    = "run_${NumClients}clients_${NumRounds}rounds${BatchSuffix}"
        $RunDir      = "$BatchDir\$RunLabel"
        New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

        $NumMalicious    = [math]::Floor($NumClients * $AttackFrac)
        $NumHonest       = $NumClients - $NumMalicious
        $GanacheAccounts = $NumClients + 5

        Write-Host ""
        Write-Host "============================================================"
        Write-Host "  EXPERIMENT $CurrentRun/$TotalRuns : $NumClients clients, $NumRounds rounds"
        Write-Host ('  ({0} honest, {1} malicious)' -f $NumHonest, $NumMalicious)
        Write-Host "============================================================"

        $RunStart = Get-Date

        # ── Step 1: Cleanup ──────────────────────────────────
        Write-Host "[1/6] Cleaning up old processes..."
        Stop-ByPattern "client.py"
        Stop-ByPattern "server.py"
        Stop-ByPattern "attack_simulator.py"
        Stop-ByPattern "ganache"
        Stop-OnPort 7545
        Stop-OnPort 9090
        Start-Sleep -Seconds 2
        Write-Host "  Done."

        # ── Step 2: Start Ganache ────────────────────────────
        Write-Host ""
        Write-Host ('[2/6] Starting Ganache (port 7545, {0} accounts)...' -f $GanacheAccounts)
        $ganacheProc = Start-Process powershell.exe `
            -ArgumentList "-ExecutionPolicy","Bypass","-File","`"$GancheScript`"","--port","7545","--accounts",$GanacheAccounts `
            -RedirectStandardOutput "$RunDir\ganache.log" `
            -RedirectStandardError  "$RunDir\ganache_err.log" `
            -NoNewWindow -PassThru

        $ganacheReady = $false
        for ($i = 1; $i -le 15; $i++) {
            Start-Sleep -Seconds 1
            if (Test-Port 7545) { $ganacheReady = $true; break }
            if ($ganacheProc.HasExited) { break }
        }
        if (-not $ganacheReady) {
            Write-Host "  ERROR: Ganache failed. Check $RunDir\ganache.log"
            $Failed++
            continue
        }
        Write-Host ('  Ganache ready (PID: {0})' -f $ganacheProc.Id)

        # ── Step 3: Deploy contracts ─────────────────────────
        Write-Host ""
        Write-Host "[3/6] Deploying smart contracts..."
        $truffleProc = Start-Process powershell.exe `
            -ArgumentList "-ExecutionPolicy","Bypass","-File","`"$TruffleScript`"","migrate","--reset","--network","development" `
            -RedirectStandardOutput "$RunDir\truffle.log" `
            -RedirectStandardError  "$RunDir\truffle_err.log" `
            -NoNewWindow -PassThru -Wait
        if ($truffleProc.ExitCode -ne 0) {
            Write-Host "  ERROR: Truffle migration failed. Check $RunDir\truffle.log"
            Stop-Process -Id $ganacheProc.Id -Force -ErrorAction SilentlyContinue
            $Failed++
            continue
        }
        Write-Host "  Contracts deployed."

        # ── Step 4: Start FL server ──────────────────────────
        Write-Host ""
        Write-Host ('[4/6] Starting FL server ({0} clients, {1} rounds)...' -f $NumClients, $NumRounds)
        $serverProc = Start-Process $PythonExe `
            -ArgumentList "server.py","$NumClients","$NumRounds" `
            -WorkingDirectory $ScriptDir `
            -RedirectStandardOutput "$RunDir\server.log" `
            -RedirectStandardError  "$RunDir\server_err.log" `
            -NoNewWindow -PassThru

        $serverReady = $false
        for ($i = 1; $i -le 20; $i++) {
            Start-Sleep -Seconds 1
            if (Test-Port 9090) { $serverReady = $true; break }
            if ($serverProc.HasExited) { break }
        }
        if (-not $serverReady) {
            Write-Host "  ERROR: Server failed to start. Check $RunDir\server.log"
            Stop-Process -Id $ganacheProc.Id -Force -ErrorAction SilentlyContinue
            $Failed++
            continue
        }
        Write-Host ('  Server ready (PID: {0})' -f $serverProc.Id)

        # ── Step 5: Launch clients ───────────────────────────
        Write-Host ""
        Write-Host ('[5/6] Launching {0} clients...' -f $NumClients)
        $clientProcs = @()

        for ($i = 0; $i -lt $NumHonest; $i++) {
            $pArgs = @("client.py","$i","$NumClients")
            if ($NoDropout) { $pArgs += "--no-dropout" }
            $p = Start-Process $PythonExe -ArgumentList $pArgs `
                -WorkingDirectory $ScriptDir `
                -RedirectStandardOutput "$RunDir\client_$i.log" `
                -RedirectStandardError  "$RunDir\client_${i}_err.log" `
                -NoNewWindow -PassThru
            $clientProcs += $p
        }

        if ($NumMalicious -gt 0 -and $AttackType -ne "none") {
            for ($i = $NumHonest; $i -lt $NumClients; $i++) {
                $p = Start-Process $PythonExe `
                    -ArgumentList "attack_simulator.py","$i","$AttackType","$NumClients" `
                    -WorkingDirectory $ScriptDir `
                    -RedirectStandardOutput "$RunDir\client_$i.log" `
                    -RedirectStandardError  "$RunDir\client_${i}_err.log" `
                    -NoNewWindow -PassThru
                $clientProcs += $p
            }
        }
        Write-Host ('  All {0} clients launched.' -f $NumClients)

        # ── Wait for server (max 5 min) ──────────────────────
        $MaxWait  = 300
        $WaitStart = Get-Date
        Write-Host "  Waiting for training (timeout: ${MaxWait}s)..."

        while (-not $serverProc.HasExited) {
            $elapsed = (Get-Date) - $WaitStart
            if ($elapsed.TotalSeconds -ge $MaxWait) {
                Write-Host "  TIMEOUT: killing experiment..."
                Stop-Process -Id $serverProc.Id -Force -ErrorAction SilentlyContinue
                $clientProcs | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
                $Failed++
                break
            }
            Start-Sleep -Seconds 2
        }

        # ── Step 6: Collect results ──────────────────────────
        Write-Host ""
        Write-Host "[6/6] Collecting results..."
        if (Test-Path "results") {
            Copy-Item "results\*" -Destination $RunDir -Recurse -Force -ErrorAction SilentlyContinue
            Write-Host "  Results saved to: $RunDir\"
        } else {
            Write-Host "  WARNING: No results directory found!"
        }

        # Save tail of server log as session report
        $serverLog = Get-Content "$RunDir\server.log" -ErrorAction SilentlyContinue
        if ($serverLog) {
            $serverLog | Select-Object -Last 80 | Set-Content "$RunDir\session_report.txt"
        }

        # Cleanup
        $clientProcs | ForEach-Object { Stop-Process -Id $_.Id -Force -ErrorAction SilentlyContinue }
        Stop-Process -Id $ganacheProc.Id -Force -ErrorAction SilentlyContinue
        Stop-ByPattern "client.py"
        Stop-ByPattern "attack_simulator.py"
        Stop-ByPattern "server.py"
        Start-Sleep -Seconds 3

        $RunTime = [math]::Round(((Get-Date) - $RunStart).TotalSeconds)
        $Completed++
        Write-Host ""
        Write-Host "  DONE $CurrentRun/$TotalRuns : $NumClients clients x $NumRounds rounds (${RunTime}s)"
    }
}

# ── Final cleanup ─────────────────────────────────────────────
Stop-ByPattern "ganache"
Stop-ByPattern "client.py"
Stop-ByPattern "attack_simulator.py"
Stop-ByPattern "server.py"

$BatchTime = [math]::Round(((Get-Date) - $BatchStart).TotalSeconds)
$BatchMins = [math]::Floor($BatchTime / 60)
$BatchSecs = $BatchTime % 60

Write-Host ""
Write-Host "╔══════════════════════════════════════════════════════════╗"
Write-Host "║              ALL EXPERIMENTS COMPLETE                   ║"
Write-Host "╠══════════════════════════════════════════════════════════╣"
Write-Host "║  Completed : $Completed / $TotalRuns runs"
Write-Host "║  Failed    : $Failed runs"
Write-Host "║  Total time: ${BatchMins}m ${BatchSecs}s"
Write-Host "╚══════════════════════════════════════════════════════════╝"
Write-Host ""
Write-Host "  Results: $BatchDir\"
Write-Host ""
