param([int]$Port=8765,[string]$Root="data/v32",[string]$Config="",[switch]$EnableCloud)
$repo=Split-Path -Parent $PSScriptRoot; Set-Location $repo
$env:PYTHONPATH=Join-Path $repo "src"; $py=Join-Path $repo ".venv-gate24\Scripts\python.exe"; if(-not(Test-Path $py)){$py="python"}
$args=@('-m','nmind.cli','bot','ui','--root',(Join-Path $repo $Root),'--port',$Port)
if($Config){$args += @('--config',$Config)}
if($EnableCloud){$env:NMIND_ENABLE_CLOUD='1'} else {$args += '--offline'}
& $py @args
