param([string]$Root="data\v32")
$repo=Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH=Join-Path $repo "src"
$py=Join-Path $repo ".venv-gate24\Scripts\python.exe"
if (-not (Test-Path $py)) {$py="python"}
& $py -m nmind.cli bot stop --root (Join-Path $repo $Root)
