$ErrorActionPreference = "Stop"

# -------------------------
# Paths
# -------------------------
$RepoRoot   = "C:\Users\ACER\Documents\GitHub\Fake-SHA"
$BackendDir = Join-Path $RepoRoot "backend"
$PythonExe  = Join-Path $BackendDir ".venv\Scripts\python.exe"
$Port       = 8000

if (-not (Test-Path $PythonExe)) {
  throw "Python venv not found at: $PythonExe"
}

function Stop-ListenerOnPort {
  param([int]$TargetPort)

  $pids = @()
  try {
    $pids = @(Get-NetTCPConnection -LocalPort $TargetPort -State Listen -ErrorAction SilentlyContinue |
      Select-Object -ExpandProperty OwningProcess -Unique)
  } catch {
  }

  if ($pids.Count -eq 0) {
    $netstat = netstat -ano | Select-String ":$TargetPort\s"
    foreach ($line in $netstat) {
      if ($line -match "\s+(\d+)\s*$") {
        $pids += [int]$Matches[1]
      }
    }
    $pids = @($pids | Select-Object -Unique)
  }

  foreach ($procId in $pids) {
    if ($procId -le 0) { continue }
    if ($procId -eq $PID) { continue }
    try {
      $proc = Get-Process -Id $procId -ErrorAction Stop
      Write-Host "Stopping $($proc.ProcessName) (PID $procId) on port $TargetPort..."
      Stop-Process -Id $procId -Force -ErrorAction Stop
    } catch {
      Write-Host "Could not stop PID $procId : $_"
    }
  }

  if ($pids.Count -gt 0) {
    Start-Sleep -Seconds 1
  }
}

Stop-ListenerOnPort -TargetPort $Port

# -------------------------
# Start backend
# -------------------------
Set-Location $BackendDir

$env:FAKE_SHA_ANALYZER = "xlmr"
$env:ENABLE_SHAP = "true"
$env:SHAP_MAX_WORDS = "400"
$env:SHAP_TOP_K = "20"

# Wider, more realistic confidence (avoids clustering around 89–90%)
$env:FAKE_SHA_XLMR_TEMPERATURE = "6.0"
$env:FAKE_SHA_XLMR_CONF_CAL_STRENGTH = "0.9"
$env:FAKE_SHA_XLMR_CONF_MARGIN_WEIGHT = "0.65"
$env:FAKE_SHA_XLMR_CONF_FLOOR = "0.52"
$env:FAKE_SHA_XLMR_CONF_CAP = "0.97"

Write-Host "Starting backend at http://127.0.0.1:$Port"
& $PythonExe -m uvicorn main:app --host 127.0.0.1 --port $Port
