# N-MIND V3.1 集成报告

## 已完成

复用了 ChatService、BotRuntime、MemoryService V2、Practice、CapabilityTracker 和现有 WebBotServer；新增 V3.1 Facade API、一次真实本地 Oracle 学习周期、记忆写入/纠正/删除 API、增长聚合、权限状态、V3.1 Windows 脚本和集成测试。

一次学习不是假状态：它从 Practice 生成新算术题，执行独立程序工具并调用 verify，向现有 CapabilityTracker 写入 tool/deterministic_oracle 记录，并在 EventLog 保存证据。不会增加 neural 成绩。

## 实际边界

- 网络请求：0
- optimizer.step：0
- 外部 LLM：未使用
- Stable checkpoint：未修改
- Memory V2 60任务成长实验：NOT_RUN
- GUI手工/浏览器自动化：NOT_RUN；真实HTTP后端和SQLite路径已实现并由自动化测试覆盖

## 启动

在仓库执行 scripts/v31_doctor.ps1，再执行 scripts/v31_start.ps1 -Port 8765，打开 http://127.0.0.1:8765/。V3.1 默认离线，CPU模式可用。

## D1-D5

D2、D4、D5 已完成离线真实服务验证；D1聊天沿用现有会话持久化与 ChatService，API 已接入但本轮未进行浏览器手工操作；D3 API 已接入 Memory V2，跨重启和删除语义沿用既有 Memory V2 测试，完整人工演示未执行。未运行的项目明确记录为 NOT_RUN/PARTIAL，未用模拟数据冒充通过。

## 测试

V3/V3.1 定向测试：6 passed。全仓回归：257 passed，3 skipped，0 failed，1 warning，exit code 0。警告为既有 PyTorch tensor 转标量提示。
