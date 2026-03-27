# ══════════════════════════════════════════════════════════════
#   Secure Federated Learning — Windows Setup Script
#   Usage: .\setup.ps1
# ══════════════════════════════════════════════════════════════

Write-Host ""
Write-Host "============================================================"
Write-Host "  Secure FL — Windows Setup"
Write-Host "============================================================"

# ── 1. Check Node.js ─────────────────────────────────────────
Write-Host ""
Write-Host "[1/4] Checking Node.js..."
try {
    $nodeVer = node --version 2>&1
    Write-Host "  Node.js found: $nodeVer"
} catch {
    Write-Host "  ERROR: Node.js not found."
    Write-Host "  Install from: https://nodejs.org  (LTS version)"
    Write-Host "  Or run:  winget install OpenJS.NodeJS.LTS"
    exit 1
}

# ── 2. Install Ganache + Truffle ──────────────────────────────
Write-Host ""
Write-Host "[2/4] Installing Ganache and Truffle..."
try {
    $ganacheVer = ganache --version 2>&1
    Write-Host "  Ganache already installed: $ganacheVer"
} catch {
    Write-Host "  Installing Ganache..."
    npm install -g ganache
}

try {
    $truffleVer = truffle version 2>&1 | Select-Object -First 1
    Write-Host "  Truffle already installed: $truffleVer"
} catch {
    Write-Host "  Installing Truffle..."
    npm install -g truffle
}

# ── 3. Check Python / conda env ───────────────────────────────
Write-Host ""
Write-Host "[3/4] Checking Python environment..."
$pythonExe = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pythonExe) {
    Write-Host "  ERROR: python not found in PATH."
    Write-Host "  Activate your conda environment first:  conda activate myenv"
    exit 1
}
$pyVer = python --version 2>&1
Write-Host "  Python: $pyVer  ($pythonExe)"

# ── 4. Install Python packages ────────────────────────────────
Write-Host ""
Write-Host "[4/4] Installing Python dependencies..."
pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: pip install failed. Check requirements.txt."
    exit 1
}

Write-Host ""
Write-Host "============================================================"
Write-Host "  Setup complete!"
Write-Host "  Before running experiments, always activate your env:"
Write-Host "    conda activate myenv"
Write-Host ""
Write-Host "  Then run:"
Write-Host "    .\run.ps1                          # default run"
Write-Host "    .\run.ps1 -AttackType label_flip -AttackFrac 0.2"
Write-Host "    .\run_experiments.ps1"
Write-Host "============================================================"
Write-Host ""
