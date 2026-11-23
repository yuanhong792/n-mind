# V3 架构

UI/API → `V3Facade` → 现有 Chat/Memory/Knowledge/Runtime；任务和事件分别由 `TaskService`、`EventLog` 持久化。自主循环先产生有界目标，再调用已授权的规则/Oracle练习，记录来源和资源消耗；任何神经训练仍须另行授权。
