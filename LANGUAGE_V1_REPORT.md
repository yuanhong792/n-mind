# LANGUAGE V1 实施报告

状态：已完成离线基础实现与零步前置验证；正式神经训练为 `PENDING_USER_APPROVAL`。

本轮补齐了依赖-free 的可逆 Byte-Level BPE 候选（4096/8192 均可配置），保留既有 `TokenizerV2` 和历史 checkpoint 兼容路径。BPE 使用全部 256 个字节作为基础字母表，未见中文、emoji、代码和标点均可无损解码；词表和 merge 表带 SHA 校验。

已复用现有随机初始化 `NMindLM`、语料来源登记、清洗和文档级切分。训练、验证、封存必须由已批准来源提供，封存集不得进入训练。

## 证据

- `experiments/language_v1/LANGUAGE_V1_PREFLIGHT.json`：随机初始化模型前向，`optimizer_steps=0`。
- `tests/test_language_v1_foundations.py`：未见 Unicode 往返、词表序列化、数据切分测试。
- 任务书附件在本地文本末尾被截断于“**P”，因此 P0/P1 的完整规模与验收细则无法从附件证实；未自行补写缺失要求。

## 未执行

未下载公开语料、未调用外部模型或 API、未修改 stable checkpoint、未执行 `optimizer.step`。CPU/GPU正式训练、性能矩阵和候选模型比较待独立训练授权及完整数据/许可登记后执行。
