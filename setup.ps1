param(
    [switch]$SkipPipInstall,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

function Write-Step($msg) {
    Write-Host "[setup] $msg" -ForegroundColor Cyan
}

function Get-VenvPythonPath {
    return Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
}

Write-Step "Project root: $PSScriptRoot"

$venvPython = Get-VenvPythonPath
if (-not (Test-Path $venvPython)) {
    Write-Step "Creating virtual environment (.venv)..."
    & $PythonExe -m venv (Join-Path $PSScriptRoot ".venv")
} else {
    Write-Step "Virtual environment already exists."
}

$venvPython = Get-VenvPythonPath
if (-not (Test-Path $venvPython)) {
    throw "Could not find venv python at $venvPython"
}

if (-not $SkipPipInstall) {
    Write-Step "Upgrading pip..."
    & $venvPython -m pip install --upgrade pip

    $req = Join-Path $PSScriptRoot "requirements.txt"
    if (Test-Path $req) {
        Write-Step "Installing requirements..."
        & $venvPython -m pip install -r $req
    } else {
        Write-Step "requirements.txt not found, skipping pip install."
    }
} else {
    Write-Step "Skipping pip install (--SkipPipInstall)."
}

Write-Step "Checking imports..."
& $venvPython -c "import torch; print('torch:', torch.__version__)"
& $venvPython -c "import cv2; print('opencv:', cv2.__version__)"

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "To activate the environment manually:"
Write-Host "  .\.venv\Scripts\activate"
Write-Host ""
Write-Host "Quick checks:"
Write-Host "  .\.venv\Scripts\python -m src.train --help"
Write-Host "  .\.venv\Scripts\python -m src.realtime_webcam"
