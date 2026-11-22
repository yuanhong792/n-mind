# Gate-2.1 失败诊断

## 结论

本轮没有新增训练，也没有修改封存集或 stable checkpoint。高置信主因是训练数据覆盖和损失定义，而不是先验地把失败归因给 N-Cell 或 Workspace。

1. `gate2_trainer.train_arm()` 使用 `rows = items[:256]`。Gate-2 训练集顺序是 L0 的 256 条 followed by L1 的 512 条，因此实际三臂都只训练 L0，L1 的位置、移动和事实更新样本为 0 条。
2. 训练目标对完整 `prompt + answer` 做普通 next-token CE，没有答案 span mask。loss 大部分来自复制题面 prompt，标签置乱仍能降低 loss；这解释了 loss 下降与语义回答失败可以同时出现。
3. 评测使用自由生成，方向正确；teacher-forced 分数只能作为辅助，不能替代自由生成。当前 evaluator 的 byte fallback 对非法单字节输出标记为 `<invalid-byte>`，不会把它伪装成中文答案。
4. 随机模型的 full-vs-state-chunked 诊断用于检查递归 state 协议；它不调用 optimizer.step，也不把 state 诊断当作语言能力。

之前关于“源码中文字节损坏”的判断已撤回：直接检查 codepoint 后，生成器与 oracle 的中文 codepoint 正确；Windows 终端显示乱码不是数据损坏证据。

## 诊断命令

```powershell
$env:PYTHONPATH = "src"
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m pytest -q tests/test_language_v2_gate21.py
```

已有 Gate-2 checkpoint 仅作为诊断输入，不能被重新训练或晋升。修复方向应是：预登记并修复完整 L0/L1 采样、增加答案 span mask、单独记录 prompt loss/answer loss，并重新建立零训练基线后再申请新的训练预算。
