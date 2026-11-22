# ONLINE V1 Real Run Report

## Gate-1 实际结果（2026-11-22）

用户已明确批准本阶段后，已完成一次串行真实采集：4 次 GET（robots.txt + 3 个固定页面），4/4 返回 HTTP 200，重定向 0，未授权请求 0。robots SHA-256 为 `ddf15edaeea82d5097df61da8e6f59ccdf66de0c20820f1367d0d076f5a9fd63`；页面原始响应累计 236,883 bytes，单页最大 132,338 bytes。知识库已写入 3 documents、165 chunks、1 host，全部 READY，信任级别 `public-read-only`。查询 `range 5` 已命中 `controlflow.html` 并返回来源 URL/片段。optimizer.step=0，stable checkpoint 未修改。Gate-2/3 和 30 题评测仍为 NOT_RUN。

日期：2026-11-22（Asia/Shanghai）

## Gate-0

通过。正确 Windows 解释器为 `C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe`；Python 3.11.9；PyTorch 2.11.0+cpu；CUDA 不可用（`torch.cuda.is_available() == False`，设备数 0）。默认 `python` 是隔离工具解释器，缺少 torch，未再使用。

命令：

```powershell
& 'C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_bot_v1_web.py -q -k test_web_site_authorization_flow
& 'C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests/test_safe_transport.py tests/test_bot_v1_web.py tests/test_bot_v1_gui.py tests/test_knowledge_v1.py tests/test_idle_explorer.py -q
& 'C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe' -m pytest tests -q
```

结果：定向 Gate-0 通过；离线网络/授权/知识/UI/探索测试通过；全仓 pytest 通过（100%，无失败），仅 1 个既有 PyTorch warning。

## 公网实验状态

`NOT_RUN` / `PENDING_USER_APPROVAL`。未发送 DNS、HTTP 或 HTTPS 请求；未访问 `docs.python.org`。授权范围、预算和三页 Gate-1 URL 已登记，但站点保持 disabled，`network_enabled=false`。

## 硬约束证据

- optimizer.step：0；stable checkpoint：未修改；外部 LLM：0；网页命令执行：0。
- 未授权请求：0；私网请求：0；真实公网页面：0。
- 30 题评测：`NOT_RUN`，封存题目未改动，未使用模拟结果代替。

## 下一步门槛

只有用户明确批准 `docs.python.org`、路径范围和预算后，才可将授权状态改为 approved 并执行 Gate-1；本报告不构成批准。
