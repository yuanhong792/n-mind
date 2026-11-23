# N-MIND Language V1.1 Gate-0 报告

状态：Gate-0 CPU 前置通过；Gate-1 保持 `PENDING_USER_APPROVAL`。

## 实际环境

本次完整 pytest 使用 `.venv-gate24\Scripts\python.exe`，结果为 **276 passed, 3 skipped, 1 warning**。系统默认 `python` 缺少 torch，未用于验收。隔离环境为 Python 3.11.9、PyTorch 2.15.0+cpu，CUDA 不可用。

## BPE-N-Cell 集成

Byte-Level BPE 词表为 4096；随机初始化 LanguageV2Model 的 embedding 形状为 `[4096, 32]`，LM head 形状为 `[4096, 32]`，输出为 `[1,T,4096]`。中文、英文、emoji、代码和标点通过 BPE 后进入 `tok_emb`，前向完成。

## 反向审计

Gate-0 只执行 `loss.backward()`，未创建 optimizer，`optimizer.step=0`。报告文件 `experiments/language_v1/LANGUAGE_V1_1_GATE0.json` 保存各模块 grad norm。N-Cell、Embedding、Head、Workspace、block_to_ws 的梯度应以实际 JSON 为准；当前原始稀疏路由路径若 Router 梯度为 0，必须视为已知限制，不能宣称 Router 学到了语言任务。

## Candidate 与隔离

已生成独立 `candidate_gate0.pt` 并完成保存/加载 SHA 复核；它不是 Stable，未覆盖 Stable。诊断关闭了训练态负载统计 buffer，避免把运行计数器误报为权重变化。Router 当前硬 Top-K 路径未收到语言 Loss 梯度，这是后续 Gate-1 前必须明确处理的架构限制。

## Gate-1

预登记文件为 `experiments/language_v1/GATE1_PRE_REGISTRATION.json`，主训练 32 步、标签置乱 32 步、总计 64 步、墙钟 600 秒。未获得独立明确授权前不执行 `optimizer.step`。GPU 计划为 `NOT_RUN_GPU_UNAVAILABLE`。
