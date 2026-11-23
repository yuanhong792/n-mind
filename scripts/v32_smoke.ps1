$ErrorActionPreference="Stop"
$repo=Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH=Join-Path $repo "src"
$py=Join-Path $repo ".venv-gate24\Scripts\python.exe"
if (-not (Test-Path $py)) {$py="python"}
& $py -c "from pathlib import Path; from nmind.v3 import V3Facade; f=V3Facade(Path('data/v32-smoke')); print(f.status()); print(f.learn_once()); print(f.growth())"
