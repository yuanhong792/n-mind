param([int]$Port = 8765, [string]$Root = "data\v31")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repo "src"
$py = Join-Path $repo ".venv-gate24\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Write-Host "Starting offline local workbench at http://127.0.0.1:$Port/"
& $py -m nmind.cli --offline bot ui --root (Join-Path $repo $Root) --port $Port
