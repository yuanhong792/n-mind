# Windows 使用

```powershell
powershell -ExecutionPolicy Bypass -File scripts\v34_start.ps1 -Port 8765
Start-Process http://127.0.0.1:8765/
```

Use the “知识与长期经验” panel for query/status. Structured clients can call `/api/v34/knowledge/entities`, `/api/v34/knowledge/sources/import-local`, `/api/v34/knowledge/assertions`, `/api/v34/knowledge/query`, and `/api/v34/experiences`.
