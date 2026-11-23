# N-MIND V3 本地启动

V3 工作台默认绑定 `127.0.0.1`，网络默认关闭，神经训练在本版本硬禁用。

```powershell
Set-Location D:\脉脑\cpu本地模型开发\cpu本地模型开发\n-mind
.\scripts\v3_doctor.ps1
.\scripts\v3_start.ps1 -Port 8765
```

浏览器打开 `http://127.0.0.1:8765/`。V3 API 包括 `/api/v3/status`、`/api/v3/chat`、`/api/v3/tasks`、`/api/v3/events` 和 `/api/v3/learn`。

`/api/v3/learn` 只运行有界的规则/Oracle练习，返回 `optimizer_steps=0`；它不代表神经模型学会了新能力。
