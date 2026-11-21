# 决策记录（docs/decisions.md）

> 只记**已发生的**架构取舍与小袁明确确认的事项。建议≠决策。

## D1 采用"路由稀疏块 + 全局工作空间"，暂缓 FastCell/SlowCell 双时间尺度

- 日期：2026-11-21（前一会话定骨架，本会话写入文档）
- 内容：最小可训练原型用 8 个独立权重 N-Cell + Top-K 路由 + 竞争写入工作空间；双时间尺度列为 M3 后消融。
- 依据：提示词允许按实验调整，但要求新意由消融体现；双时间尺度在 depth=1 基线建立前引入会分不清收益来源。

## D2 训练目标为 5+1 项联合损失（CE+pred+var+balance+entropy+probe）

- 日期：2026-11-21（前一会话实现，本会话实测其副作用）
- 现状：**联合损失被怀疑拖累决策学习**——同架构稠密版用纯 CE 72 步遮挡段 0.933，联合损失稀疏版 720 步仅 0.61（experiments/2026-11-21_m1_overfit.json + results/baselines.json）。
- 后续：把两项变量拆开重跑，再决定降权或重设计；在此之前不扩大模型规模。

## D3 默认 demo 配置训练预算不足，保留但在文档标注

- 事实：默认 6 epoch（≈36 步）遮挡段 0.459，多数类基线 0.444——接近无信息水平。
- 处理：不悄悄改默认值（改配置=改复现口径），在 ROADMAP/STATUS 标注"欠训练"，M2 收口时一并定新默认。

## D4 torch 先用 CPU 版，GPU 轮子待小袁确认

- 事实：本机 torch 2.10.0+cpu；RTX 5070（Blackwell）需 cu128 轮子，约 3GB 下载。
- 决策：所有实验先按 CPU 口径做（这也是项目验收目标）；装 GPU 轮子属环境变更，等小袁点头。

## D5 评测层修复采用"两键自适应"而非统一键名

- 日期：2026-11-21
- 内容：`fit()/score()` 同时接受 `logits`（GRU/SharedTrunk）与 `action_logits`（NModel），不强迫任一侧改接口。
- 理由：NModel 的输出契约被 trainer/engine/测试多处依赖，改动面大；自适应最小且显式。

## D6 pred_mode 默认翻转为 "next"（2026-11-21，M2.2 验证后）

- 决策：`ModelCfg.pred_mode` 默认由 "current" 改为 "next"；legacy "current" 保留为消融开关。
- 依据：legacy 目标把要预测的状态当输入（z_hat[t]=f(z[t],a[t-1])，MSE(z_hat[t],z[t])），实测学成恒等拷贝（cos(z_hat,z)=0.998、动作敏感性 0.025、未来位置读出不超拷贝基线），三臂 world_model_validated=false；next 模式（(s_t,a_t)→s_{t+1}，detach 目标）在 F10_next 两 seed validated=true（敏感性 1.05-1.15、z_hat 单独读出下一位置 0.92-0.93 超真值等价读出 R2），主任务不退化（0.91-0.97）。证据：experiments/m2_2_validation/{pred_eval,pred_rollout}.json。
- 兼容：旧 checkpoint 重载需 allow_config_mismatch=True（形状不变，pred 语义不同；决策头不读 z_hat，动作输出不受影响）。load_checkpoint 按存档 cfg 重建结构，_model_kwargs 已补记 router_ste/pred_mode（此前漏记导致语义静默回退，已修）。
- 不翻转 router_ste 默认：它改变训练动力学且 C 臂（任务盲路由对照）依赖其关闭状态，保持显式 opt-in（F10/F10_next 配置）。
- 已知代价：multi-step rollout 解码位置准确率绝对值仍低（h10 0.19-0.31，读出天花板 0.96），远未达到"可靠世界模型"；本决策只确立目标语义正确，不宣称能力达标。
