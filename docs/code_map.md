# docs/code_map.md — N-MIND R1 代码地图（2026-11-21，M2.2 会话实测路径）

> 按实际文件写，不臆造。所有路径相对仓库根 `D:\cpu本地模型开发\n-mind`。

## 核心（src/nmind/）

| 文件 | 内容 | 本轮改动 |
|---|---|---|
| `core/architecture.py` | NModel：encoder→路由→NCell 块→workspace→predictor→head 的单步图；`forward_sequence` 时序展开 | +`router_ste`、+`pred_mode`（"current"/"next"）、step 接受 `action_cur` |
| `core/ncell.py` | NCell 门控循环块；`forward`（稠密）与 `forward_sparse`（稀疏 bmm，数学等价） | forward_sparse 的 weights 分支改为 mask-STE（前向=硬采纳，router 梯度=h_upd−h_sel） |
| `core/router.py` | BlockRouter Top-K 路由 + 负载均衡偏置（buffer `load_bias`） | 未改 |
| `core/workspace.py` | GlobalWorkspace 固定槽位竞争读写 | 未改 |
| `core/encoder.py` | EventEncoder：数值/文本→z | 未改（上一轮修过设备 bug） |
| `core/predictor.py` | WorldPredictor + normalized_error/cosine_error/方差正则 | 未改（pred_mode 改动在 architecture 层） |
| `core/decision_head.py` | ActionHead（ctx+w+mem→logits） | 未改 |
| `learning/trainer.py` | `train()`/`train_continual()`/`evaluate()`/`fit_posthoc_probe()`/`make_data()` | +`router_ste`/`pred_mode` 参数贯通，compute_total pred_shift |
| `learning/losses.py` | LossWeights（含 detach_pred_target）+ compute_total | +`pred_shift` 对齐（next 模式 z_hat[:, :-1]→z[:, 1:]） |
| `learning/datasets.py` | GridTask/v2-disjoint-pairs split/oracle（v2 为并行 Codex 会话引入） | 未改 |
| `learning/checkpoint.py` | save/load（含 probe、config_hash） | 未改 |
| `config.py` | pydantic 配置（ModelCfg/RuntimeCfg/LearningCfg…） | ModelCfg +`pred_mode: Literal["current","next"]="current"` |
| `cli.py` | doctor/demo/train/evaluate/memory/benchmark/baseline/ablate/cloud | 未改 |
| `language/micro_lang.py` | M3.1 微型语言生成器（oracle、系统性 OOD） | 未改（本轮新增测试已钉住） |
| `runtime/engine.py` | NMindEngine 交互闭环（记忆+治理） | 未改 |
| `bridge/` `memory/` | 云端策略门 / SQLite 情景记忆 | 未改 |

## 实验（experiments/）

| 目录/文件 | 内容 |
|---|---|
| `2026-11-21_m2_2x2.py` | 2×2+I+FIX 消融 runner（run_arm 可复用；**FIX 无 F10_ste**，F10 走专用脚本） |
| `m2_ablation/` | 全部 checkpoint（A-D/I/F1-F8/F9/F10 × seeds）+ fix/fix2/ms3/iso JSON + leak_audit |
| `m2_closeout/` | reload_eval（22ckpt Δ=0）、gradient_diagnostics、cpu_bench、f9_f10、registry、报告与渲染器 |
| `m2_2_validation/` | **本轮**：paired_seeds_preregister（预登记 seed）、run_paired（配对+统计）、pred_eval（世界模型显式测试）、gradient_intervention（轨迹干预）、environment.json |
| `m3_language/` | run_m3_v1.py + m3_v1_results.json + checkpoint |
| `2026-11-21_m1_overfit.py` | M1 过拟合验收 |

## 测试（tests/，52 项全绿 @ 本轮）

test_ncell / test_router / test_workspace / test_datasets / test_router_ste /
test_closeout_regressions / test_micro_lang / test_pred_next_mode（本轮新增）

## 已知口径注意

- `statistics`：本轮起 std 统一 **ddof=1**（样本）；此前报告 0.914±0.031 为 ddof=0。
- `run_arm`（m2_2x2.py）的 flags 直接传 `train(**flags)`，支持 `router_ste=True`；
  但其 FIX 字典不含 F10 —— F10 配对实验用 `m2_2_validation/run_paired.py`。
- pred 分支 legacy 语义缺陷（目标当输入）由 `pred_eval.py` 量化，修复走
  `pred_mode="next"`（默认 current 不变，旧 checkpoint 复现口径不受影响）。
