# AGENTS.md — N-MIND R1 项目开发约束

任何会话（Codex / Claude Code / WorkBuddy / AutoClaw）进入本仓库，先读本文件，再读 `ROADMAP.md` 与 `STATUS.md`。

## 一、不可变原则（违反即停）

1. **从零训练**：所有可训练权重随机初始化。禁止加载 Qwen / Llama / Gemma / DeepSeek / SmolLM 等任何预训练权重、词嵌入、教师模型或蒸馏输出。公开原始训练数据与基础计算库（PyTorch / NumPy）可用。
2. **本地优先**：核心推理、记忆、评测必须断网可跑。云端调用默认关闭（`cloud.mode: off`），任何联网都要显式配置密钥并记录审计；绝不把第三方 API 伪装成本地能力。
3. **CPU 必须可跑**：加载、推理、评测不得依赖 CUDA。GPU 是可选加速（当前 torch 为 CPU 版，CUDA 路径未启用，见 STATUS）。
4. **诚实记录**：所有指标来自实际执行。没跑就写"未运行"，没证明就写"研究假设"。禁止伪造准确率 / 步数 / 完成状态；禁止改测试集掩盖失败。
5. **不写密钥**：配置只记环境变量名（`NMIND_CLOUD_API_KEY`），绝不写值。数据库、聊天记录、用户隐私不入库。
6. **增量学习不静默重建**（2026-11-22）：续训必须从已验证 checkpoint 逐位继承权重（`train(model=...)` / `fit_existing_model`），语义冲突报错而非重建；冻结参数逐位不变（torch.equal）；批量重训/扩 seed 须用户批准（AdaptationBudget 硬限制）；三种运行模式 inference / memory_only / neural_adapt 分开计量，训练态 buffer（load_bias/block_load_count）不冒充学习。

## 二、环境（本机实测，2026-11-21）

| 项 | 值 |
|---|---|
| Python | 3.12.10（`C:\Users\Administrator\AppData\Local\Programs\Python\Python312\python.exe`） |
| torch | **2.10.0+cu128**（2026-11-21 升级，RTX 5070 可用；默认设备 auto→cuda；CPU 路径必须继续可用，见 AGENTS 第三节） |
| 硬件 | i7-14700KF 28 逻辑核 / 32GB RAM / RTX 5070 12GB（torch CPU 版暂用不上） |
| 安装 | 仓库根目录 `pip install -e .` 后用 `python -m nmind.cli ...` |
| 测试 | pytest 9.2.0，72 项（含 router_ste/pred_mode/micro_lang/H3b-H4b/true_continual 回归） |

## 三、常用命令

```powershell
python -m nmind.cli doctor                 # 硬件/依赖/配置体检
python -m nmind.cli demo --json            # M1：训练→保存→重载→逐位一致校验
python -m nmind.cli train --epochs 40 --json
python -m nmind.cli train --continual --json   # A→B→重测 A（持续学习）
python -m nmind.cli evaluate --json        # 任务套件（需先有 checkpoint）
python -m nmind.cli benchmark --json       # CPU 延迟/吞吐/RSS
python -m nmind.cli baseline --json --epochs 12
python -m nmind.cli ablate --json          # 结构/记忆/云端消融
python -m pytest tests -q
```

- `--json` 把结果写到 `results/*.json`：**结论必须有对应 JSON 证据**。
- `--seed` 固定随机种子；同 seed 同权重（checkpoint 保存/加载逐位一致已在 demo 验收）。
- `--offline` 硬断网。

## 四、代码约定

- src 布局：`src/nmind/{core,learning,evaluation,memory,runtime,bridge,language}`；配置用 pydantic，未知键直接报错（防止"以为开了其实没开"）。
- 文件 UTF-8（无 BOM）。中文注释解释"为什么"，不写"做了什么"。
- 消融开关内建在 `NModel` / `train()` 参数里，基线与主模型跑同一份代码路径，不靠外部脚本改行为。
- 每个实验 = `experiments/` 里一个可运行脚本 + 一个 JSON 结果，README.md 索引。
- 数据生成器标签必须程序可计算（oracle），训练/测试集 RNG 流不相交（`seed*1000+1/+2`）。
- 数值异常不许静默：非有限损失跳过并计数（`non_finite_steps_skipped`）。

## 五、禁止事项

- 禁止下载任何预训练模型权重（`language.auto_download` 恒为 false）。
- 禁止在文档里把"建议 / 假设 / 计划"写成"已完成"。
- 禁止把大文件（权重、DB、日志）提交 Git：见 `.gitignore`。
- 禁止清理或覆盖用户目录里的既有文件。
- 禁止声称"会话结束后仍能后台自动运行"。
