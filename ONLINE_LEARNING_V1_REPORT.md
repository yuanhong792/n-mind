# ONLINE LEARNING V1 Report

日期：2026-11-22（Asia/Shanghai）

## 执行边界

本轮已启动源码审计与离线安全测试；未访问公网、未提交真实域名或请求计划、未启用真实网络、未执行 Gate-B 005，未调用外部 LLM/预训练模型/教师模型。

## 复用与实现盘点

- `BotRuntime` / `BotPolicy`：默认 `network_enabled=false`、候选训练关闭、optimizer 步数预算为 0。
- `safe_transport`：HTTPS、默认 443、userinfo/IP/本地域名拦截、DNS 全量公有地址校验、固定解析地址连接、逐跳重定向重新校验、MIME/响应大小/解压大小/超时/429/403/重定向上限。
- `SourceRegistry`：用户显式授权站点、域名边界和路径边界、策略哈希审计。
- `KnowledgeStore` / `KnowledgeSearch`：独立 SQLite/WAL 数据库、版本保留、内容哈希去重、中文 bigram + FTS5/LIKE 检索、引用字段和低信任状态。
- `IdleWebExplorer`：有界队列、同授权站点链接、深度/日配额/单轮配额/cooldown、用户消息优先中断。
- `ChatService` / UI：`neural`、`retrieval`、`tool` 三路分离；展示来源、链接、信任状态、检索/错误说明。
- `experience_bridge`：仅允许 verified 且有可信反馈来源的神经候选行；网页资料不直接进入神经训练。

## 本轮修复

1. 真实安全传输现在拒绝缺失或不在允许集合中的 `Content-Type`，避免二进制/归档响应被文本提取器误接收。
2. 空闲探索每个任务开始前再次检查日配额，避免一个批次在剩余配额不足时继续发起请求。
3. 用户主动抓取路径在连接前检查日配额；配额耗尽时记录审计事件并明确返回“未发起请求”。

## 测试与证据

- `python -m compileall -q src`：通过。
- 全量 `python -m pytest tests -q`：未进入测试执行。当前 Windows Python 环境缺少 `torch`，22 个测试模块在 collection 阶段因 `ModuleNotFoundError: torch` 失败；没有伪造通过状态。
- `tests/test_safe_transport.py` 与 `web_security_probe.py` 已具备注入 resolver/connector 的离线安全探针，真实 socket 发送断言为 false；本轮未联网运行。

## 未完成/需授权事项

真实公网 30 篇以内授权网页采集、30 道独立题检索与引用验收，必须在用户先批准具体域名、路径、请求数量/字节/时间计划后进行；本轮不宣称完成。由于 torch 缺失，神经权重零修改只能由源码策略和后续可运行环境的权重哈希测试共同确认。

## 失败案例记录

- 全量测试失败：环境依赖缺失 `torch`，属于测试环境阻断。
- 未授权/离线请求：按设计返回 `blocked`，记录 `offline_mode`、`network_disabled` 或配额原因，不创建真实连接。
- 跨站重定向、私网/回环/混合 DNS、非 HTTPS、userinfo、非默认端口、二进制 MIME：按设计拒绝并记录结构化错误码。
