$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repo "src"
$py = Join-Path $repo ".venv-gate24\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Write-Host "Python: $py"
& $py -c "import sys,sqlite3; print(sys.version); print('sqlite=',sqlite3.sqlite_version)"
& $py -c "import torch; print('torch=',torch.__version__); print('cuda=',torch.cuda.is_available())"
Write-Host "Network default: disabled; training: disabled; optimizer.step: 0"
& $py -m pytest tests/v3 -q
