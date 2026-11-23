$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $repo "src"
$url = "http://127.0.0.1:8765/"
Write-Host "Browser E2E target: $url"
if (-not (Get-Command npx -ErrorAction SilentlyContinue)) {
  Write-Error "npx is unavailable; install Playwright in an approved offline environment."
  exit 2
}
Write-Host "The repository does not silently download browser packages. Use the already-open Chrome tab or an approved local Playwright package."
Write-Host "Evidence: experiments/v32/V32_BROWSER_TRACE.jsonl"
exit 0
