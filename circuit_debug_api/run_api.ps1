param(
  [switch]$RefreshModel
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Resolve-Path $Root)

$PyCandidates = @(
  ".\\.venv312\\Scripts\\python.exe",
  "..\\.venv312\\Scripts\\python.exe"
)
$Py = $null
foreach ($candidate in $PyCandidates) {
  if (Test-Path $candidate) {
    $Py = $candidate
    break
  }
}
if (-not $Py) {
  $Py = "python"
}

if (-not (Test-Path ".\\assets\\model_bundle.joblib")) {
  Write-Host "Assets not found. Building runtime assets..."
  & $Py .\\build_runtime_assets.py
}

# Build hybrid assets (LoRA adapter + KNN reference/index) if missing.
if ($RefreshModel -or -not (Test-Path ".\\assets_hybrid\\hybrid_config.json")) {
  if ($RefreshModel) {
    Write-Host "Rebuilding hybrid assets (auto-select best current model)..."
  } else {
    Write-Host "Hybrid assets not found. Building hybrid assets..."
  }
  & $Py .\\build_hybrid_assets.py
}

& $Py -m uvicorn server:app --host 127.0.0.1 --port 8001
