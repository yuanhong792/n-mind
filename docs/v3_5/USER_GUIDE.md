# Windows 使用

```powershell
powershell -ExecutionPolicy Bypass -File scripts\v35_start.ps1 -Port 8765
Start-Process http://127.0.0.1:8765/
```

在“多步骤安全行动”面板输入目标，点击“预览计划”，检查步骤和权限范围后点击“批准计划”，最后点击“执行计划”。不批准时后端不会执行任何动作。
