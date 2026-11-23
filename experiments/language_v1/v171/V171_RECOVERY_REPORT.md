# V1.7.1 Checkpoint 抢救报告

## 结论（先看这里）

| 问题 | 结论 |
|---|---|
| 1536 步真实训练后的模型权重还在不在？ | **不在。FINAL_WEIGHTS_LOST** |
| V1.7 Final 究竟是不是父模型？ | **是。逐张量完全相同（L1=0.0，0 个张量变化）** |
| 是否存在 1536 步末尾权重副本？ | **不存在。已搜索项目目录与 Windows 临时目录，未发现任何优化器状态、自动保存或中间权重文件** |

## 一、文件级证据（SHA-256 与元数据）

| 模型 | SHA-256 | 步数元数据 | 逐张量 vs 父模型 |
|---|---|---|---|
| V1.5 Parent | `5681f656fbb35f4f4a033e16212897f2d7cc02eac6754b9d860fb980d6f6a611` | 256 | 基准 |
| V1.6 Baseline | `7102a19a00f4f808...` | 256 | 不同（21 张量，L1≈1163.78） |
| V1.6 Fixed Router | `174b2ba6ffd624a0...` | 256 | 不同（21 张量，L1≈1163.78） |
| V1.7 Best | `72cde8a7c1b94e1f62d9a4529d9c7e200c61266ea118a28d340093dd52ab36c3` | **0** | **完全相同（L1=0.0）** |
| V1.7 Final | `1e352f6b01273815f6ee94c6155c8f336e073a91c3a55206ccbdc1df835b1d38` | **1536** | **完全相同（L1=0.0）** |

关键点：Best 与 Final 的文件 SHA 不同，**但这仅由 `steps` 与 `saved_at` 元数据造成**。两份文件的 `state_dict` 与 V1.5 父模型逐张量、逐元素完全一致（总 L1 = 0.0，最大绝对差 = 0.0，21 个张量全部 identical，Router `score_mlp` L1 = 0.0）。文件 SHA 不能证明权重不同，本次以逐张量比较为准。

## 二、根因（代码级）

`train_v17.py` 原保存顺序（约 220–228 行）：

```python
m.load_state_dict(best_state); m.eval()      # 先把模型恢复成 best
sealed_eval = evaluate(m, sealed, ...)
final_state = {k: v.detach().clone() for k, v in m.state_dict().items()}  # ← 此刻抓到的是 best 权重
save_language_v2(..., "candidate_v17_best.pt", steps=best_step, ...)
m.load_state_dict(final_state)                # ← 又把它装回去
save_language_v2(..., "candidate_v17_final.pt", steps=step, ...)  # steps=1536 只是元数据
```

本次 `best_step = 0`（父模型即为验证集最优），因此 `best_state` = 父模型权重。`final_state` 在恢复之后才拷贝，导致「Final」实际保存的是父模型权重，只是元数据写了 `steps=1536`。**1536 步优化器更新后的真实权重从未落盘。**

## 三、恢复尝试（穷尽且失败）

审计脚本 `_checkpoint_audit.py` 以相同 tokenizer（`hash=26af116ebd13ff00`）与严格架构（`load_state_dict(strict=True)`）加载了以下全部候选：

- V1.5 Parent / V1.6 Baseline / V1.6 Fixed Router / V1.7 Best / V1.7 Final
- V1.4 Main / V1.5 QueryAttention（同架构可加载者）
- 项目内全部 `*.pt` / `*.pth` / `*.ckpt`、`*checkpoint*`、优化器状态与 autosave 命名文件
- Windows `%TEMP%` / `%TMP%` 与 `/tmp` 下全部 `*.pt`（均为 pytest 临时夹具，非 v17 训练产物）

结果：

- 未发现任何 v17 优化器状态文件（`optimizer_state_files_found = 0`）
- 未发现任何 v17 自动保存 / 中间 checkpoint（`autosave_files_found = 0`，`v17_intermediate_checkpoints_found = 0`）
- 除 V1.6 / V1.4 / V1.5-QueryAttn 这些**已知的其它实验权重**外，没有任何权重与父模型不同，也没有任何可归因于「v17 训练 1536 步末尾」的副本

## 四、处置

- 不覆盖、不改写任何现有权重文件（本轮 `no_weight_written = true`）。
- **不伪造、不反向重建**丢失的训练权重；不把 `steps=1536` 元数据当作训练证据。
- `candidate_v17_final.pt` 在后续公平评测中**按无效候选处理**（其权重=父模型，不提供任何新信息）。
- 保存逻辑缺陷已在 P1 修复（`train_v17.py`），并新增回归测试，防止再次发生。

完整逐文件 / 逐张量数据见 `V171_CHECKPOINT_AUDIT.json`。
