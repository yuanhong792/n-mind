# V3.6 实际状态与复用审计

## 当前可复用代码

- N-Cell/Router/Workspace 与 Language V2：`src/nmind/language/`
- Neural-only 与训练审计：`src/nmind/evaluation/`、`src/nmind/learning/`
- 聊天来源边界：`src/nmind/bot/chat_service.py`
- 外部 Provider 审计与策略：`src/nmind/bridge/providers.py`、`src/nmind/bridge/policy.py`
- V3.3/V3.4/V3.5 工作台、Memory、知识图谱和安全规划模块保持复用。

## 本轮实现边界

DeepSeek 通过 OpenAI-compatible Chat Completions 接口作为可选 `cloud_teacher` 来源接入。默认仍为 `cloud.mode=off`，API key 只从环境变量 `NMIND_DEEPSEEK_API_KEY` 读取；源码、YAML、SQLite、审计摘要和 Git 中不保存密钥。`--offline` 强制关闭外部调用。

外部回答不会标记为 `neural`，不会进入 Neural-only 成绩，也不会触发 optimizer.step 或 Candidate 晋升。训练数据、Teacher 调用与独立评测必须分开审计。

## 当前限制

本轮未执行公网 API 请求，未执行新神经训练；因此 DeepSeek 连通性、费用和回答质量为 `NOT_RUN`。V3.6 的神经能力仍不能由外部问答结果替代。
