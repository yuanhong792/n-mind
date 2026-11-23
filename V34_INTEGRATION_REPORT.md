# N-MIND V3.4 集成报告

## 已完成

- 新增 `KnowledgeGraphService`：实体、别名、证据、断言、时间字段、关系遍历、冲突记录和撤回。
- 新增经验记录与撤回，并将 V3.3 已验证 Practice/Oracle 周期登记为 `VERIFIED_WITHIN_SCOPE`，不修改 neural 能力账本。
- 新增 `/api/v34/knowledge/*` 与 `/api/v34/experiences` API，并把 UI 接到真实状态/查询接口。
- SQLite 使用 WAL、外键和 additive `IF NOT EXISTS` schema；Memory V2 原表未修改。
- 定向测试 11 passed；全仓 pytest 257 passed、3 skipped、0 failed、1 warning。

## 安全边界

optimizer.step=0，网络请求=0，外部模型调用=0，Stable checkpoint 只读。结构化输入不执行命令；无证据断言默认为 `UNVERIFIED`；撤回内容从正常关系查询和经验查询中排除。

## 真实限制

V34_BROWSER_E2E 当前所有 D1-D10 保留 `NOT_RUN`，因为本轮 API/SQLite 证据已完成但尚未逐项完成浏览器操作闭环。V34_RETRIEVAL_ABLATION 也为 `NOT_RUN`，没有伪造检索质量分数。旧 V3.3 未完成项目仍保持其原状态。

## 启动

```powershell
powershell -ExecutionPolicy Bypass -File scripts\v34_start.ps1 -Port 8765
Start-Process http://127.0.0.1:8765/
```
