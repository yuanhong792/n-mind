param([string]$Root = "data\v31")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repo "src"
$py = Join-Path $repo ".venv-gate24\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py -c "from nmind.bot.runtime import request_control; request_control(r'$((Join-Path $repo $Root))','stop'); print('stop requested')"
