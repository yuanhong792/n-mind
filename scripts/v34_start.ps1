param([int]$Port=8765,[string]$Root="data/v32")
$repo=Split-Path -Parent $PSScriptRoot; Set-Location $repo
$env:PYTHONPATH=Join-Path $repo "src"; $py=Join-Path $repo ".venv-gate24\Scripts\python.exe"; if(-not(Test-Path $py)){$py="python"}
& $py -m nmind.cli --offline bot ui --root (Join-Path $repo $Root) --port $Port
