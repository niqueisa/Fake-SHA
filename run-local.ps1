$ErrorActionPreference = "Stop"

# -------------------------
# Paths
# -------------------------
$RepoRoot   = "C:\Users\Isabelle\Documents\GitHub\Fake-SHA"
$BackendDir = Join-Path $RepoRoot "backend"
$PythonExe  = Join-Path $BackendDir ".venv\Scripts\python.exe"

if (-not (Test-Path $PythonExe)) {
  throw "Python venv not found at: $PythonExe"
}

# -------------------------
# Start backend
# -------------------------
Set-Location $BackendDir

$env:FAKE_SHA_ANALYZER = "xlmr"
$env:ENABLE_SHAP = "true"
$env:SHAP_MAX_WORDS = "400"
$env:SHAP_TOP_K = "20"

& $PythonExe -m uvicorn main:app --host 127.0.0.1 --port 8000