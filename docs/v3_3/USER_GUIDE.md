# V3.3 Windows 使用

```powershell
powershell -ExecutionPolicy Bypass -File scripts\v32_doctor.ps1
powershell -ExecutionPolicy Bypass -File scripts\v33_start.ps1 -Port 8765
Start-Process http://127.0.0.1:8765/
```

在“自主学习决策”面板中先启用 V3.2 的自主学习开关，再执行“按决策练习”。暂停、恢复、停止均调用真实 `/api/v33/learning/*` API。网络保持关闭，练习结果标记为 `tool/oracle`，不计入 neural 能力。
