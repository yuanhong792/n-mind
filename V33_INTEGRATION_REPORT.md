# N-MIND V3.3 集成报告

## 已完成

- 复用 V3.2 Facade、EventLog、TaskService、Practice、CapabilityTracker 和 Web 工作台。
- 新增持久化 `V33LearningEngine`：候选目标、原因码、优先级、失败记录、复习时间、权限和暂停状态均写入 SQLite。
- 新增真实 API：`/api/v33/learning/status|goals|decision-explanation|run-once|pause|resume|stop|reviews|failures|growth|events|policy`。
- Web 工作台新增自主学习决策面板，按钮调用真实后端。
- V3.3 定向测试 9 passed；全仓 257 passed、3 skipped、0 failed、1 warning。

## 安全边界

optimizer.step=0，网络请求=0，外部模型调用=0，stable checkpoint 未修改。自主学习默认关闭，必须用户显式启用；Oracle 成功只记为 `tool/oracle`，不提升 neural 能力。

## 真实限制

D1-D9 的完整浏览器逐项证据尚未全部执行，当前标记见 `experiments/v33/V33_E2E_RESULTS.json`；未运行项目不宣称通过。ROUND_ROBIN/WEAKEST_FIRST 公平策略对照尚未执行，未宣称 V3.3 优于基线。历史 276 节点清单仍不可得。

## 启动

```powershell
powershell -ExecutionPolicy Bypass -File scripts\v33_start.ps1 -Port 8765
Start-Process http://127.0.0.1:8765/
```
