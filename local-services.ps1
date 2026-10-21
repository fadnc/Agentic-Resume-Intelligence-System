param(
    [ValidateSet('start', 'stop', 'restart', 'status')]
    [string]$Action = 'start'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $root 'venv\Scripts\python.exe'
$stateDir = Join-Path $root '.local-services'
$logDir = Join-Path $root '.local-logs'
$services = @('mlflow', 'backend', 'frontend')

function Get-Health([string]$Name) {
    $url = switch ($Name) {
        'mlflow' { 'http://127.0.0.1:5001/health' }
        'backend' { 'http://127.0.0.1:8001/health' }
        'frontend' { 'http://127.0.0.1:5000/' }
    }
    try {
        $response = Invoke-WebRequest -Uri $url -TimeoutSec 2 -UseBasicParsing
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500
    } catch { return $false }
}

function Stop-ManagedServices {
    foreach ($name in @('frontend', 'backend', 'mlflow')) {
        $pidFile = Join-Path $stateDir "$name.pid"
        if (Test-Path $pidFile) {
            $savedPid = 0
            [void][int]::TryParse((Get-Content $pidFile -Raw).Trim(), [ref]$savedPid)
            if ($savedPid -gt 0) {
                $processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
                $parent = $processes | Where-Object { $_.ProcessId -eq $savedPid }
                if ($parent -and $parent.ExecutablePath -eq $python) {
                    $tree = [System.Collections.Generic.List[object]]::new()
                    $tree.Add($parent)
                    for ($index = 0; $index -lt $tree.Count; $index++) {
                        $children = $processes | Where-Object { $_.ParentProcessId -eq $tree[$index].ProcessId -and $_.ExecutablePath -eq $python }
                        foreach ($child in $children) { $tree.Add($child) }
                    }
                    for ($index = $tree.Count - 1; $index -ge 0; $index--) {
                        Stop-Process -Id $tree[$index].ProcessId -Force -ErrorAction SilentlyContinue
                    }
                    Write-Host "Stopped $name and its managed Python child processes."
                }
            }
            Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
        }
    }
}

if ($Action -eq 'stop') {
    Stop-ManagedServices
    exit 0
}
if ($Action -eq 'restart') {
    Stop-ManagedServices
    Start-Sleep -Seconds 2
}
if ($Action -eq 'status') {
    foreach ($name in $services) {
        $running = Get-Health $name
        Write-Host ("{0}: {1}" -f $name, $(if ($running) { 'running' } else { 'offline' }))
    }
    exit 0
}

if (-not (Test-Path $python)) { throw "Project Python not found: $python. Create/install the venv first." }
New-Item -ItemType Directory -Force -Path $stateDir, $logDir | Out-Null

$launches = @(
    # MLflow's default Uvicorn worker supervisor uses multiprocessing pipes,
    # which can fail with WinError 5 on some Windows setups. Waitress avoids that.
    @{ Name = 'mlflow'; Args = @('-m', 'mlflow', 'server', '--host', '127.0.0.1', '--port', '5001', '--backend-store-uri', "sqlite:///$($root -replace '\\','/')/mlflow.db", '--default-artifact-root', "$root/mlflow-artifacts", '--waitress-opts', '--threads=4') },
    @{ Name = 'backend'; Args = @('-m', 'uvicorn', 'backend.app:app', '--host', '127.0.0.1', '--port', '8001', '--workers', '1') },
    @{ Name = 'frontend'; Args = @('flask_frontend\app.py') }
)

foreach ($item in $launches) {
    $name = $item.Name
    if (Get-Health $name) {
        Write-Host "$name is already responding; leaving it running."
        continue
    }
    $pidFile = Join-Path $stateDir "$name.pid"
    if (Test-Path $pidFile) {
        $oldPid = 0
        [void][int]::TryParse((Get-Content $pidFile -Raw).Trim(), [ref]$oldPid)
        $oldProcess = if ($oldPid -gt 0) { Get-CimInstance Win32_Process -Filter "ProcessId = $oldPid" -ErrorAction SilentlyContinue } else { $null }
        if ($oldProcess -and $oldProcess.ExecutablePath -eq $python) {
            Write-Host "$name process is already running (PID $oldPid)."
            continue
        }
        Remove-Item $pidFile -Force -ErrorAction SilentlyContinue
    }
    $stdout = Join-Path $logDir "$name.out.log"
    $stderr = Join-Path $logDir "$name.err.log"
    $env:MLFLOW_TRACKING_URI = 'http://127.0.0.1:5001'
    $env:BACKEND_URL = 'http://127.0.0.1:8001'
    $proc = Start-Process -FilePath $python -ArgumentList $item.Args -WorkingDirectory $root -WindowStyle Hidden -PassThru -RedirectStandardOutput $stdout -RedirectStandardError $stderr
    Set-Content -Path $pidFile -Value $proc.Id -NoNewline
    Write-Host "Started $name (PID $($proc.Id))."
    Start-Sleep -Milliseconds 700
}
Write-Host 'Local service URLs: frontend http://localhost:5000 | API http://localhost:8001/docs | MLflow http://localhost:5001'
Write-Host "Logs: $logDir"
