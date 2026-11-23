# N-MIND V3.6 神经与外部教师接入报告

本轮完成了 DeepSeek 可选接入和神经/外部来源隔离。适配器复用既有 OpenAI-compatible Provider，新增 `provider: deepseek`，默认端点为 `https://api.deepseek.com`、默认模型为 `deepseek-chat`。

启用方式：

```powershell
$env:NMIND_DEEPSEEK_API_KEY = "<用户自己的密钥>"
python -m nmind.cli --config configs/deepseek.example.yaml bot ui --root data/v32 --port 8765
```

不要把密钥写入 YAML、源码、SQLite 或提交记录；需要断网运行时继续使用 `--offline`，它会硬性关闭 Provider。

聊天响应来源为 `cloud_teacher`，明确标注外部问答，不计入 N-Cell 神经能力。Neural-only 评测关闭 Memory、Knowledge、Tool、Teacher；外部回答不能作为神经学习成功证据。

## 本轮状态

- API 实际请求：`NOT_RUN`
- 新 optimizer.step：`0`
- Stable checkpoint：未修改
- 外部预训练权重/Embedding/API模型作为神经核心：未使用
- 定向聊天回归：通过

V3.6 的 M1 输出协议、M2 最小神经任务、M3 可塑性、M4 持续学习和 M6 内部诊断仍需独立实验与封存评测，不能用 DeepSeek 回答冒充完成。
