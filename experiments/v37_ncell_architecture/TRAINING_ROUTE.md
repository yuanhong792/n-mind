# N-MIND V3.7 免费云CPU / 本地GPU训练分流规则

**用户确认：** 免费GitHub Actions CPU能够承担的神经实验继续在GitHub云端运行；超过预设预算的训练转由用户本人Windows电脑执行。云端不支付GPU费用、不设置自托管Runner、不远程启动/访问用户电脑。

## 本轮实现

- `experiments/v37_ncell_architecture/route_training.py` 为只生成规划的路由器。输出`V37_TRAINING_ROUTE.json`和`V37_RUN_LOCAL.ps1`，**生成脚本绝不自动运行**。
- `GITHUB_FREE_CPU`：保留云端Actions执行；本项目**保守分流策略**：每臂≤128步、所有臂总步数≤768、预估峰值内存≤10GB、无CUDA需求。这些是项目自设阈值，**并非GitHub官方免费配置或精准资源预测**。
- `LOCAL_MANUAL`：超过任何阈值、明确要求CUDA或手动选择本地时，仅生成离线本地计划与配置。必须用户显式启动PowerShell脚本，且本地真实CUDA可用时才运行。不可静默切换CPU假装使用GPU。
- 本地Candidate最多每臂2048步，默认每臂3600秒墙钟硬限制；不会自动续跑、无限增量更新或晋升Stable。
- 本地命令使用`--save-candidates`，只在实验输出目录下写入`candidates/tokenizer.json`与`candidate_<arm>_seed...pt`。本轮保存**模型权重与实验摘要**，未实现跨进程优化器状态恢复，故中断的训练不能宣称无缝续训。
- 不允许DeepSeek/其他教师API、外部工具/Memory参与神经独立测试。不会上传本地私有训练数据或模型到GitHub。

## GitHub云端预演

GitHub Actions在当前PR中执行**小预算真实训练**和分流协议测试，并生成一份模拟超预算的本地清单。Actions仅输出给用户的PowerShell文件，**不能执行本地GPU训练**。

查看CI产物中的`ncell-v37-training-route`：分为`route_cloud/`与`route_local/`。后者的`V37_RUN_LOCAL.ps1`需要由用户在项目根目录手动运行。

## 在本地手动启动超预算训练

确保本地已检出本实验分支，且`python -c "import torch; print(torch.cuda.is_available())"`返回`True`。从Actions下载`ncell-v37-training-route`工件，找到`route_local/V37_RUN_LOCAL.ps1`，并在项目根目录执行：

```powershell
powershell -ExecutionPolicy Bypass -File .\route_local\V37_RUN_LOCAL.ps1
```

若安全策略禁止执行脚本，可直接审查其内容后复制内部`python experiments/v37_ncell_architecture/run_ablation.py ...`命令。不要运行来源不明的脚本。

## 分流示例

云CPU：6臂 × 128步，总计最多768个optimizer步骤，项目预算内。

转本地：6臂 × 256步，总计最多1536步，由本地CUDA手动启动；CI只生成任务文件，不代用户操作本地电脑。

**需要单独继续研发的事项：** 优化器和随机数状态断点续训、可复用的模型发布、自动识别峰值内存和运行时、GPU实际吞吐量与显存估算、多seed新鲜封存验收、结果经审批后上传GitHub。现有脚本不能自动保证大模型在12GB GPU上可运行。

## GitHub CPU Candidate 工件（2026-11-23 补强）

- 仅在现有预算内的云端 6 臂 × 128 步开发集训练中启用 `--save-candidates`，在 CI 工作空间输出 `ci_extended/candidates/tokenizer.json` 和六个独立 `candidate_*.pt`。
- CI 在上传前逐个重新加载模型，验证架构可加载、每臂 128 步、文件数与训练臂一致。工件名 `ncell-v37-cloud-candidates`，训练报告继续单独保留为 `ncell-v37-extended-dev-report`。
- GitHub 工件有保留期限；这不等于长期模型发布。仅为可下载的隔离候选产物，不自动提交二进制权重到仓库、也不替换 Stable。
- 本地离线工作流保持 `LOCAL_MANUAL`，只能用户在 Windows 主动执行。未获得新的超额训练授权，不扩大训练预算。
- 估计 RAM 若为 NaN/Inf 则直接拒绝分流，避免无效预算被误判为免费 CPU 任务。

## 远程 Windows GPU 训练（2026-11-23 授权）

远程训练需 Windows 主动安装和启动受限轮询代理；无入站端口或 GitHub 自托管 Runner，且只能训练 Candidate。请见 REMOTE_GPU.md。本地代理未连接之前，GPU 不会远程启动。
