# V3.5 复用与安全边界

复用 V3.4 KnowledgeGraph、Memory V2、V3.3 LearningEngine、TaskService、PlanningStore 和现有 WebBotServer。新增 `V35PlanStore` 与 `SafeActionRegistry`。

白名单动作仅包括本地工作区读取、知识查询、经验查询、确定性计算和结构化报告。没有 shell、Python、浏览器脚本、网络客户端或文件写入动作。所有计划先进入 `AWAITING_APPROVAL`，批准记录持久化后才能执行。
