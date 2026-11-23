param([int]$Port = 8765, [string]$Root = "data\v3")
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repo "src"
$py = Join-Path $repo ".venv-gate24\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
Write-Host "V3 starts in offline mode; network and training remain disabled."
& $py -m nmind.bot.cli ui --offline --root (Join-Path $repo $Root) --port $Port
