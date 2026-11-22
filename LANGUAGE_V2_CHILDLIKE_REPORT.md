# N-MIND Language V2：纯数据渐进学习报告

## 本轮结论

已完成离线 P0 工程闭环的源码实现：来源登记与许可门禁、离线清洗去重、可逆 Tokenizer V2、独立 LanguageWorld 情境环境、课程任务接口和显式训练权限门。默认网络关闭、正式训练关闭、optimizer.step 为 0、Gate-B 005 继续暂停，stable checkpoint 未写入。

这不是儿童语言能力验收。本轮没有获取公开语料、没有执行正式训练，也没有把检索或规则答案计入神经能力。

## 数据来源

FineWeb-2、CulturaX、Wikimedia Wikipedia 当前只登记为待审候选源，未自动批准、未联网下载。Python 文档知识库仍是 ExternalKnowledge/retrieval，不会自动进入 NeuralCandidateTraining。自建 `LanguageWorld` 数据标为程序生成、独立 oracle，不冒充公开原始语料。

## 已实现模块

- `data_sources.py`：来源、许可、撤销和 fail-closed 门禁。
- `corpus_pipeline.py`：Unicode 规范化、中文比例、敏感信息最小化、去重和 manifest。
- `world.py`：对象/颜色/位置/移动/打开及独立 oracle。
- `curriculum.py`：基础任务族和难度入口。
- `learning_v2.py`：正预算授权检查和零步参数指纹诊断。
- `tokenizer_v2.py`、`neural_language_v2.py`：上一阶段的可逆编码和随机初始化 N-Cell/Router/Workspace 模型。

## 验证状态

Gate-1 新增来源拒收、敏感信息清洗、重复/预算、情境状态转移、非法动作、课程 oracle、训练权限、人工语料分集、Tokenizer round-trip 和零训练基线测试共 17/17 通过；BotRuntime 暂停/恢复独立复验通过。随机初始化神经基线对 3 个 starter 任务实际为 0/3 exact-match。全仓回归此前出现的旧 BotRuntime 快照竞态已定位并修复，但最终全仓结果尚未重新跑完；未修改旧测试。

## 未完成/未运行

许可通过后的公开数据获取、L1-L4 正式训练、反事实神经能力、事实更新、指代、多轮自由生成、Memory V2 经验成长及联网自主探索接入均未运行。下一步必须先取得独立训练预算和具体来源许可，再使用候选 checkpoint 训练并经过 sealed、遗忘和人工晋升门禁。

## Windows 命令

```powershell
$env:PYTHONPATH = "src"
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pytest -q tests/test_language_v2.py tests/test_language_v2_pipeline.py
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pytest -q
```
