<#
.SYNOPSIS
    TuriX-Win Helper Script for OpenCLaw
    Supports dynamic task injection, resume, and skills system.

.EXAMPLE
    ./run_turix.ps1 "Open Chrome and search for TuriX"
#>

param (
    [Parameter(Position = 0)]
    [string]$Task,

    [Alias("r")]
    [string]$Resume,

    [Alias("c")]
    [string]$Config = "examples/config.json",

    [switch]$NoPlan,

    [switch]$EnableSkills,

    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

# ---------- Configuration ----------
$ProjectDir = Get-Location
$ConfigFile = Join-Path $ProjectDir $Config
$EnvName = "turix_env"

function Log-Info { Write-Host "[INFO] $args" -ForegroundColor Green }
function Log-Warn { Write-Host "[WARN] $args" -ForegroundColor Yellow }
function Log-Error { Write-Host "[ERROR] $args" -ForegroundColor Red }

# ---------- Validation ----------
if (-not $Resume -and -not $Task) {
    Log-Error "Task or -Resume required"
    exit 1
}

if (-not (Test-Path $ConfigFile)) {
    Log-Error "Config not found: $ConfigFile"
    exit 1
}

# ---------- Update Config ----------
function Update-Config {
    Log-Info "Updating configuration..."
    
    $json = Get-Content $ConfigFile -Raw | ConvertFrom-Json
    
    if ($Task) {
        $json.agent.task = $Task
    }
    
    if ($Resume) {
        $json.agent.resume = $true
        $json.agent.agent_id = $Resume
    }

    if ($NoPlan) {
        $json.agent.use_plan = $false
        $json.agent.use_skills = $false
    }

    if ($EnableSkills) {
        $json.agent.use_skills = $true
        if (-not $json.agent.skills_max_chars) {
            $json.agent.skills_max_chars = 4000
        }
    }

    $json | ConvertTo-Json -Depth 10 | Set-Content $ConfigFile -Encoding UTF8
    Log-Info "Config updated successfully"
}

# ---------- Main ----------
function Main {
    if ($DryRun) {
        Log-Info "[DRY RUN]"
        Log-Info "  Task: $(if ($Task) { $Task } else { "(resume) $Resume" })"
        Log-Info "  Config: $ConfigFile"
        Log-Info "  Command: conda run -n $EnvName python examples/main.py"
        return
    }

    Update-Config

    Log-Info "Starting TuriX..."
    Log-Info "Press Ctrl+Shift+2 to force stop"

    # Execute TuriX
    conda run -n $EnvName python examples/main.py
}

Main
