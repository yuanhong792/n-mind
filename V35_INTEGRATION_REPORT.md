# N-MIND V3.5 集成报告

## 已完成

- 复用 V3.4 图谱、Memory V2、V3.3 学习引擎、TaskService 和 PlanningStore。
- 新增 2–5 步 V3.5 计划预览，步骤包含依赖、工具、验证器、幂等键和最大尝试次数。
- 新增 `V35PlanStore` 持久化计划、审批状态和恢复数据。
- 新增 `SafeActionRegistry`：工作区文件读取、知识查询、经验查询、确定性计算、结构化报告。
- 未审批执行硬返回 `PERMISSION_BLOCKED`；审批后真实执行 API/SQLite 闭环已验证。
- Web 工作台新增“多步骤安全行动”预览/批准/执行面板。

## 安全边界

optimizer.step=0，网络请求=0，外部模型调用=0，Stable checkpoint 未修改。不存在任意 Shell/Python/网页脚本或用户文件写入工具；路径越界被拒绝。

## 真实结果

API 演示“计算 2+3 并生成报告”：2 步计划，未审批为 `PERMISSION_BLOCKED`，审批后两步均 `VERIFIED_SUCCESS`。依赖不足的知识任务会阻断，不会编造结果。

全量 D1-D12 浏览器逐项验收尚未全部执行，状态以 `experiments/v35/V35_E2E_RESULTS.json` 为准；未运行场景保持 `NOT_RUN`。

## 启动

```powershell
powershell -ExecutionPolicy Bypass -File scripts\v35_start.ps1 -Port 8765
Start-Process http://127.0.0.1:8765/
```
