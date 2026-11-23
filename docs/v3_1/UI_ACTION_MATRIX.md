# UI 操作矩阵

|操作|后端|真实副作用|状态|
|---|---|---|---|
|聊天|V3Facade.chat_reply|ChatService会话/事件|已接入API|
|一次学习|V3Facade.learn_once|Practice Oracle、能力台账、事件|已接入API|
|启用/暂停自主学习|set_learning|权限事件|已接入API|
|保存记忆|memory_create|Memory V2 SQLite|已接入API|
|纠正/删除记忆|memory_correct/memory_delete|tombstone及新版本|已接入API|
|任务创建/状态|TaskService|v3_tasks.db及事件|已接入API|
|成长|growth|真实tracker/cycle聚合|已接入API|
