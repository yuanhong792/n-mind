# V3.3 安全与证据

- optimizer.step: 0；training_enabled: false；Stable checkpoint: read-only。
- network_enabled: false；决策引擎没有网络调用路径。
- 用户开关、暂停、恢复、停止均写入 EventLog；目标、复习、失败记录写入 `v33_learning.db`。
- Oracle 结果只更新 tool/oracle 账户；不会更新 neural-only 分数。
- 所有演示和验收都必须保留 PASS/PARTIAL/NOT_RUN，不以空状态冒充成功。
