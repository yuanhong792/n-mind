# ONLINE V1 Gate-2 Report

## Gate-2 实际结果

已获得明确批准并完成真实采集：robots 策略核验通过，新增7页全部 HTTP 200、无重定向、无越权请求；累计真实 GET 为13/40，累计知识库为10篇文档、641个中文片段，新增响应累计660,525 bytes。

聊天验收10题已执行：9道范围内问题全部返回 `retrieval` 并有Top-3来源依据，范围外问题返回 `none/abstain`；证据支持率10/10，范围外拒答1/1。跨重启重新打开SQLite后仍为10篇文档、641个片段，异常处理查询可命中。

optimizer.step=0，stable checkpoint未修改，外部LLM调用为0。Gate-3及30题正式验收尚未执行。

## 状态

`PENDING_USER_APPROVAL`。本轮只完成离线核验、候选页面筛选、预算登记和本地集成测试准备；没有访问新增网页。

## Gate-1 证据核验

当前工作区存在 Gate-1 审计、指标、报告和 SQLite 知识库：3篇文档、165个片段、5次真实GET记录、robots策略核验通过。Gate-1页面不重复下载。

## Gate-2 候选页

新增候选7页均为 `https://docs.python.org`，路径均落在已批准的 `/zh-cn/3.11/tutorial/` 或 `/zh-cn/3.11/library/` 范围内，且与Gate-1的3页不重复：数据结构、模块、输入输出、异常、类、标准库、pathlib文件操作。

## 预算

Gate-1已使用5次GET，理论剩余35次；Gate-2预计新增8次GET（运行时robots核验+7页），累计页面目标10篇。单页512KiB、总量12MiB、超时10秒、间隔5秒、并发1、optimizer.step=0。

## 等待事项

必须取得用户对新增7页真实访问的明确批准后，才能执行Gate-2。未批准前，网络请求数为0，状态保持PENDING_USER_APPROVAL。
