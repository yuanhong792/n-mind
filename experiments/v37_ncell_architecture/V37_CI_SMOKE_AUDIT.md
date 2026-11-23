# N-Cell V3.7 — GitHub CI 实测审计（首轮）

## 可复核出处

- 基础master：`3ec5fd702ced72116bf01919e05b44e9939c5318`
- 实测代码提交：`cc15a4c5608aba337b1952d3a00163a08e9162b9`
- [GitHub Actions Run #45132262019](https://github.com/yuanhong792/n-mind/actions/runs/45132262019)
- [完整原始JSON和自动生成Markdown报告](https://github.com/yuanhong792/n-mind/actions/runs/45132262019/artifacts/14798202804)
- GitHub环境：ubuntu-latest，Python 3.12，CPU PyTorch；并非本地Windows/RTX 5070实测

## 实际执行结果

| 项目 | 状态 | 证据 |
|---|---|---|
| V3.7结构＋既有神经定向测试 | PASS | CI步骤 `Architecture and prior neural regression tests` |
| 全仓pytest回归 | PASS | CI步骤 `Full repository tests` |
| 训练脚本+报告生成 | PASS | CI步骤 `Minimal real neural-only ablation and report` |
| JSON与Markdown上传 | PASS | Artifact `ncell-v37-ablation-report` |
| 6臂完整对照 | NOT_RUN | 仅运行legacy和router_ste两个臂 |
| 新语义封存集评测 | NOT_RUN | 未使用`--final-sealed` |
| 多随机种子实验 | NOT_RUN | 仅一个seed |
| 神经持续学习与遗忘 | NOT_RUN | 不在本轮冒烟范围内 |
| Windows/RTX 5070本地训练 | NOT_RUN | CI为Linux CPU |

执行命令：

```powershell
python experiments/v37_ncell_architecture/run_ablation.py --steps 1 --batch-size 2 --eval-samples 2 --arms legacy router_ste --threads 1 --max-seconds-per-arm 90 --output-dir experiments/v37_ncell_architecture/out
```

**真实数据：**

| Arm | 参数量 | 实际optimizer步骤 | Dev完整答案Exact |
|---|---:|---:|---:|
| `legacy` | 7,324 | 1 | 0/2 |
| `router_ste` | 7,324 | 1 | 0/2 |

这组数据仅验证训练、梯度审计、自由生成、JSON/Markdown报告和CI工件链路；1步训练、2条Dev样本无法证明结构收益。**不允许据此声称N-Cell智力提升、泛化改善或已解决MINLEARN-2的Byte非法UTF-8问题。**

## 代码与实验限制

- 输入为受限字符级合成位置绑定任务；构造96个双实体场景，按场景分为60/20/16组train/dev/sealed。每个场景包含两条相反的实体查询。
- 训练只监督答案与EOS，dev评测使用模型自由生成（无答案泄露）；封存集默认关闭，避免反复调参。
- `router_ste`仍是被选中Top-K块上的近似梯度，不等于全路由可微。
- 重置门和查询注意力引入额外参数，正式比较需要参数量匹配或独立声明差异。
- 不保存Stable、不使用DeepSeek/其他教师、不读取Memory/Tool作为答案。
- CI代码曾对Markdown字面反引号使用无效Python转义，后续单独修复，属于报告输出格式问题；训练结果不受影响。

## 正式六臂试验运行方式（仍待在本地执行）

```powershell
python experiments/v37_ncell_architecture/run_ablation.py --steps 16 --batch-size 2 --eval-samples 16 --arms legacy router_ste attention router_attention reset all --threads 4 --max-seconds-per-arm 240 --output-dir experiments/v37_ncell_architecture/local_run_01
```

此命令会在每个臂进行最多16次Candidate优化（最多96步），**只有用户本地实际执行后才有结果**。建议先运行4～8步检测环境与耗时再扩大。该对照的开发集数据只是早期探索；若使用sealed必须保持最终评估协议，不能循环打开sealed来调参。

**后续改进项：** 多seed、参数匹配、反事实配对覆盖全部评测组、训练后重启持久性、逐阶段神经状态追踪、MINLEARN-2 Byte协议单独修复和更丰富的OOV实体泛化。
