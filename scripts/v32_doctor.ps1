$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repo "src"
$py = Join-Path $repo ".venv-gate24\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py -c "import sys,torch,sqlite3; print(sys.executable); print(torch.__version__); print('cuda=',torch.cuda.is_available()); print('sqlite=',sqlite3.sqlite_version)"
& $py -m pytest --collect-only -q
