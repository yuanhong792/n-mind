# N-MIND V3.7 Windows GPU 远程训练接入

用户已授权允许远程操作 Windows GPU，仅限 N-MIND 训练。

原有 GitHub 免费 CPU 训练规则继续有效。超出免费 CPU 项目预算的训练，或者明确需要 CUDA 的训练，可通过 Windows 本地代理主动拉取任务。

安全限制：

- GitHub 公开仓库不使用 self-hosted Actions Runner，也不对外开放 Windows 端口。
- 代理仅允许六种 N-Cell Candidate 训练架构，固定 Python 训练入口，不接受任何 shell 字符串。
- 每臂最多 2048 步、任务最多 12288 步、每臂最长 3600 秒；显存不足或 CUDA 缺失直接拒绝。
- 任务指定 40 位源代码 commit SHA，本地 Git checkout 必须完全匹配且不存在追踪文件修改。
- 任务结果和 Candidate 权重只保存在 Windows 本地用户目录，绝不自动上传或升级 Stable。
- GitHub 分支拥有写权限的账户一旦被盗，可能触发受预算约束的训练。建议 Windows 使用独立低权限账户、关闭多余服务，不保存个人密码。
- 取消任务：提交 action=cancel 且 target_request_id 对应当前请求。运行时约每 180 秒拉取一次取消状态；也可在当前输出目录建立 STOP 文件。

## Windows 第一次连接

1. 在 N-MIND 电脑本地准备好 Python、CUDA PyTorch，并检查：

~~~powershell
python -c "import torch; print(torch.cuda.is_available())"
~~~

2. 本地 Git 仓库需切换到任务指定的准确 commit SHA。代理不会替你更新仓库或运行来自互联网的未审核代码。
3. 在 N-MIND 项目根目录手动执行一次检查：

~~~powershell
python scripts/nmind_gpu_remote_agent.py --once
~~~

4. 需要持续允许远程训练时，在同一目录启动轮询代理：

~~~powershell
python scripts/nmind_gpu_remote_agent.py --poll-seconds 600
~~~

5. 远程控制文件是 experiments/v37_ncell_architecture/REMOTE_GPU_REQUEST.json。默认禁用。只有明确写入 enabled=true 和具体经过预算验证的 job 才能启动训练。
6. 训练完成后在用户目录 N-MIND-remote-runs 中查看日志、REMOTE_GPU_STATUS.json 和 Candidate 权重。当前不自动上传权重。

Windows 没有运行代理之前，GitHub 更新远程任务不会启动电脑。后台进程须由用户在本地保持运行，用户可以随时结束进程。

项目公开分支的请求文件不应包含任何敏感信息或令牌。
