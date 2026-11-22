# N-MIND · Agent Runtime R1 —— Windows 一键复现（真实命令，非说明稿）
#
#   powershell -ExecutionPolicy Bypass -File experiments\bot_runtime_r1\reproduce.ps1
#
# 跑完会得到：R1 单元测试、五阶段端到端实验、九份证据文件、安全探针 PASS/FAIL。
# 默认离线、不自动更新神经权重；只有脚本里显式带 --i-authorize 的那一步会做候选训练，
# 且步数硬上限 48（任务书 §9 允许 ≤120 步 / ≤300 s）。stable 源权重只被复制，绝不被写。
#
# 注意：Step 2 会重写 experiments\bot_runtime_r1\ 下的九份证据文件（覆盖为本次实测值）。

$ErrorActionPreference = "Stop"
# Windows PowerShell 默认按 GBK 解控制台输出，会把 Python 的 UTF-8 中文显示成乱码。
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$Repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Repo
$env:PYTHONPATH = Join-Path $Repo "src"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONHASHSEED = "0"

$Stable = Join-Path $Repo "experiments\m5_language\m52_s0_checkpoint.pt"
$Tok    = Join-Path $Repo "experiments\m5_language\tokenizer_m52.json"
$Root   = Join-Path $Repo "results\bot_r1_repro"          # 手工步骤用独立目录，不动实验证据
$Py     = "python"

function Py-Cli([string[]]$ArgsList, [string]$Label) {
    Write-Host "`n=== $Label ===" -ForegroundColor Cyan
    Write-Host (">> " + ($ArgsList -join " "))
    & $Py -X utf8 @ArgsList
    if ($LASTEXITCODE -ne 0) { throw "步骤失败：$Label (exit=$LASTEXITCODE)" }
}

function Read-Status() {
    $p = Join-Path $Root "status.json"
    if (-not (Test-Path $p)) { return $null }
    try { return (Get-Content $p -Raw -Encoding UTF8 | ConvertFrom-Json) } catch { return $null }
}

function Wait-Status([scriptblock]$Pred, [int]$TimeoutSec = 30) {
    $end = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $end) {
        $s = Read-Status
        if ($s -and (& $Pred $s)) { return $s }
        Start-Sleep -Milliseconds 400
    }
    return (Read-Status)
}

if (-not (Test-Path $Stable)) { throw "缺少 stable 源权重：$Stable（M5.2 产物，不随本任务提交）" }
if (Test-Path $Root) { Remove-Item -Recurse -Force $Root }
New-Item -ItemType Directory -Force -Path $Root | Out-Null

# ---------------------------------------------------------------- Step 0 依赖体检
Py-Cli @("-m", "nmind.cli", "bot", "doctor", "--root", $Root,
         "--import-stable", $Stable, "--import-tokenizer", $Tok, "--profile", "safe-local") `
        "bot doctor（依赖/线程夹紧/权重哈希/策略/锁）"

# ---------------------------------------------------------------- Step 1 单元测试
Py-Cli @("-m", "pytest", "tests/test_bot_runtime_r1.py", "-o", "addopts=", "-q") `
        "pytest —— 任务书 §9 十五条 + §9.16 实验回归"

# ---------------------------------------------------------------- Step 2 五阶段端到端
Py-Cli @("experiments\bot_runtime_r1\run_bot_r1_experiment.py") `
        "端到端实验驱动 A–E（约 2 分钟，产出九份证据）"

# ---------------------------------------------------------------- Step 3 无人值守自主练习（独立目录，手工重跑）
Py-Cli @("-m", "nmind.cli", "bot", "once", "--offline", "--cycles", "20",
         "--root", $Root, "--import-stable", $Stable, "--import-tokenizer", $Tok) `
        "bot once —— 零用户输入时自主跑 20 轮 ×3 可验证任务"

Py-Cli @("-m", "nmind.cli", "bot", "eval", "--root", $Root) `
        "bot eval —— 封存集自由生成 EM（神经侧真实分数，预期 0.0）"

# ---------------------------------------------------------------- Step 4 常驻 + 用户交错 + 跨进程控制
Write-Host "`n=== bot run（后台常驻，profile=demo，75 s 后自停）===" -ForegroundColor Cyan
$RunLog = Join-Path $Root "run_resident.txt"
$proc = Start-Process -FilePath $Py `
    -ArgumentList @("-X", "utf8", "-m", "nmind.cli", "bot", "run", "--root", $Root,
                    "--profile", "demo", "--deadline", "75", "--poll", "1") `
    -WorkingDirectory $Repo -RedirectStandardOutput $RunLog -PassThru -NoNewWindow
Start-Sleep -Seconds 6
Write-Host ("启动后（无消息）状态：" + (Read-Status).state + " / cycles=" + (Read-Status).idle_cycles_done)

Py-Cli @("-m", "nmind.cli", "bot", "ask", "你好，N-MIND，现在你在忙什么？", "--root", $Root) `
        "bot ask #1 —— 用户消息优先处理"
$null = Wait-Status { param($s) $s.user_queue_len -eq 0 } 30

Py-Cli @("-m", "nmind.cli", "bot", "pause", "--root", $Root) "bot pause（写 pause.flag，跨进程）"
$atPause = (Wait-Status { param($s) $s.state -eq "PAUSED" } 20).idle_cycles_done
Start-Sleep -Seconds 6
$stillPaused = Read-Status
Write-Host ("暂停 6 s 后 cycles=" + $stillPaused.idle_cycles_done + "（应与 " + $atPause + " 相同）")
if ($stillPaused.idle_cycles_done -ne $atPause) { throw "pause 没有真的冻住自主练习" }

Py-Cli @("-m", "nmind.cli", "bot", "ask", "暂停期间这条消息应该先攒着，恢复后优先处理", "--root", $Root) `
        "bot ask #2 —— 暂停期间进队列，不丢"
Py-Cli @("-m", "nmind.cli", "bot", "resume", "--root", $Root) "bot resume"
$afterResume = Wait-Status { param($s) $s.state -ne "PAUSED" -and $s.user_queue_len -eq 0 } 30
Write-Host ("恢复后 cycles=" + $afterResume.idle_cycles_done + " queue=" + $afterResume.user_queue_len)

Py-Cli @("-m", "nmind.cli", "bot", "stop", "--root", $Root) "bot stop（写 stop.flag，安全边界退出）"
$proc | Wait-Process -Timeout 45 -ErrorAction SilentlyContinue
$lockPath = Join-Path $Root "lock.json"
Write-Host ("进程退出 rc=" + $proc.ExitCode + " / lock.json 残留=" + (Test-Path $lockPath))

# ---------------------------------------------------------------- Step 5 显式授权候选更新
$StableCopy = Join-Path $Root "stable_model.pt"
$ShaBefore = (Get-FileHash $StableCopy -Algorithm SHA256).Hash.ToLower()
Py-Cli @("-m", "nmind.cli", "bot", "candidate", "--i-authorize", "--steps", "48", "--root", $Root) `
        "bot candidate —— 四臂对照 + 48 步硬预算（单独授权，非默认行为）"

Write-Host "`n=== bot approve（故意不带 --i-approve，应被拒绝）===" -ForegroundColor Cyan
& $Py -X utf8 -m nmind.cli bot approve --root $Root     # 期望：promoted=false，理由写在没有可晋升候选/需人工旗标
Write-Host ("approve 退出码=" + $LASTEXITCODE + "（上面输出的 promoted 字段才是判据）")
$ShaAfter = (Get-FileHash $StableCopy -Algorithm SHA256).Hash.ToLower()
Write-Host ("stable 副本 sha256 变更前=" + $ShaBefore.Substring(0,16) + "… 变更后=" + $ShaAfter.Substring(0,16) + "…")
if ($ShaBefore -ne $ShaAfter) { throw "候选流程改动了 stable（任务书 §8 禁止）" }

Py-Cli @("-m", "nmind.cli", "bot", "audit", "--limit", "20", "--root", $Root) `
        "bot audit —— 世系、隔离区、经验计数"
Py-Cli @("-m", "nmind.cli", "bot", "logs", "--limit", "12", "--root", $Root) "bot logs"
Py-Cli @("-m", "nmind.cli", "bot", "status", "--root", $Root) "bot status"

# ---------------------------------------------------------------- Step 6 结论
$sec = Get-Content (Join-Path $Repo "experiments\bot_runtime_r1\security_tests.json") -Raw -Encoding UTF8 | ConvertFrom-Json
$summ = Get-Content (Join-Path $Repo "experiments\bot_runtime_r1\summary.json") -Raw -Encoding UTF8 | ConvertFrom-Json
Write-Host "`n================ 结论 ================" -ForegroundColor Green
Write-Host ("安全探针：PASS " + $sec.passed + " / FAIL " + $sec.failed)
Write-Host ("自主练习：" + $summ.phaseA.cycles + " 轮 / " + $summ.phaseA.attempts +
            " 次尝试 / 族 " + ($summ.phaseA.families -join ","))
Write-Host ("候选判定：" + $summ.phaseD.status + "（steps " + $summ.phaseD.steps_used_total +
            " / new_skill_gain " + $summ.phaseD.new_skill_gain + "）")
Write-Host ("资源：CPU " + $summ.phaseB.resource_totals.cpu_percent_of_machine +
            " % / RSS 峰值 " + $summ.phaseB.resource_totals.rss_peak_mb + " MB")
Write-Host ("证据目录：experiments\bot_runtime_r1\ ；报告 BOT_R1_REPORT.md")
if ($sec.failed -gt 0) { exit 1 }
Write-Host "全部步骤完成。" -ForegroundColor Green
