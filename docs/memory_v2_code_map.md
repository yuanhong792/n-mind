# Memory V2 code map

现有 `MemoryStore`、`MemoryRetriever`、`Consolidator` 保留不变并继续负责 SQLite、FTS5、纠正/冲突、巩固与删除墓碑。

新增 `nmind.memory.service.MemoryService` 作为统一门面：`observe` 写入情景/事实/程序性元数据，`recall` 做有界召回与重要性/可信度重排，`feedback` 记录反馈，`consolidate` 委托既有巩固器，`forget` 委托既有不可逆删除。

新增 schema v2 的 `memory_meta`、`memory_feedback`、`procedures` 表；这些表与 `training_runs` 分离，不能触发 optimizer 或 checkpoint 写入。`ChatService` 通过可选 `memory_service` 接入，并将结果标记为 `memory`，不冒充 `neural`。
