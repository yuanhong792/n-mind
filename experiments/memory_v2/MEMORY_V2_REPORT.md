# N-MIND Memory V2 阶段报告

## 本轮结论

已完成可运行的 V2 记忆服务层和聊天接入，未训练神经权重、未修改 stable checkpoint、未调用外部模型/API。现有 `MemoryStore`、`MemoryRetriever`、`Consolidator` 保留并复用；数据库采用 schema v2 加法迁移。

## 已实现

- 跨会话、跨重启的 SQLite 记忆保存与召回。
- 情景/语义事实/程序性/偏好等类型元数据；重要性、可信度、来源、验证状态、过期字段。
- 有界 FTS5 + 中文字符重叠回退召回；按相关性、重要性和可信度重排。
- 用户反馈提升/降低重要性；用户删除继续使用既有墓碑机制，巩固和重写不能恢复已删除内容。
- 敏感凭据类内容默认拒绝持久化。
- `ChatService` 可选接入 `MemoryService`，响应源单独标记为 `memory`，并返回 `memory_hits` 与说明，不冒充 `neural`。
- `memory_feedback`、`procedures` 表为后续经验反思和程序性经验登记预留，仍不触发 optimizer.step。

## 测试

本轮新增 4 个测试全部通过；既有聊天回归 14 个通过。完整全仓 pytest 需在交付前再跑一次并把实际计数写入本文件。

## 60 任务验收状态

尚未运行。`preregistration.json` 已封存四臂定义，但 `baseline.json` 与 `ablation_results.json` 保持 `NOT_RUN`，不以开发测试替代未见任务实验。下一步应在独立封存任务集上执行 no_memory / episodic_only / full_memory / wrong_memory，并报告任务成功、召回、跨重启、删除泄漏和负迁移。

## 能否改善未见任务

当前代码具备可验证的机制：成功任务可写入程序性记忆，新任务可按相关性召回，且来源独立可审计；但在 60 道独立任务实验实际运行前，不能声称已证明泛化提升。
