$ErrorActionPreference = "Stop"


# -------------------------
# Paths (edit only if needed)
# -------------------------
$RepoRoot   = "C:\Users\Isabelle\Documents\GitHub\Fake-SHA"
$BackendDir = Join-Path $RepoRoot "backend"
$PythonExe  = Join-Path $BackendDir ".venv\Scripts\python.exe"


if (-not (Test-Path $PythonExe)) {
  throw "Python venv not found at: $PythonExe"
}


# -------------------------
# Start backend in background window
# -------------------------
Start-Process powershell -ArgumentList @(
  "-NoProfile",
  "-WindowStyle", "Hidden",
  "-Command",
  @"
Set-Location '$BackendDir'
`$env:FAKE_SHA_ANALYZER='xlmr'
`$env:ENABLE_SHAP='true'
`$env:SHAP_MAX_WORDS='250'
`$env:SHAP_TOP_K='12'
`$env:SHAP_MAX_EVALS='64'
`$env:SHAP_CACHE_ENABLED='true'
`$env:SHAP_CACHE_MAXSIZE='256'
& '$PythonExe' -m uvicorn main:app --host 127.0.0.1 --port 8000
"@
) | Out-Null


Start-Sleep -Seconds 3


# -------------------------
# Start ngrok in background window
# -------------------------
Start-Process powershell -ArgumentList @(
  "-NoProfile",
  "-WindowStyle", "Hidden",
  "-Command",
  "ngrok http 8000"
) | Out-Null


# -------------------------
# Read ngrok public URL
# -------------------------
Write-Host "Waiting for ngrok tunnel..." -ForegroundColor Cyan
$publicUrl = $null


for ($i = 0; $i -lt 40; $i++) {
  Start-Sleep -Seconds 1
  try {
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:4040/api/tunnels" -UseBasicParsing
    $https = $resp.tunnels | Where-Object { $_.proto -eq "https" } | Select-Object -First 1
    if ($https) {
      $publicUrl = $https.public_url
      break
    }
  } catch {
    # keep waiting
  }
}


if ($publicUrl) {
  Set-Clipboard -Value $publicUrl
  Write-Host ""
  Write-Host "Backend is running." -ForegroundColor Green
  Write-Host "ngrok URL: $publicUrl" -ForegroundColor Green
  Write-Host "Copied to clipboard. Paste this into extension Backend URL setting." -ForegroundColor Yellow
} else {
  Write-Host "Could not detect ngrok URL. Check if ngrok is installed/running." -ForegroundColor Red
}
