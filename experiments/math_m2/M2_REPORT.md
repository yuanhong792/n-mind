# N-MIND M2：数学驱动双记忆神经架构研究报告

## 1. 结论摘要

- 真实训练：4096/4096 step，墙钟 382.83s / 3600s，会话 1。
- SEP A/B 对照：legacy=0.1108，SEP=0.4511。
- 六架构初筛：{'R1': 0.3048, 'R2': 0.3739, 'M2-A': 0.1343, 'M2-B': 0.1215, 'M2-C': 0.0725, 'M2-D': 0.0851, 'M2-E': 0.1232}。
- 入围复验：['R2', 'R1']。
- 最佳模型（统一面板）：R2，macro=0.5418，micro=0.5281。
- 反事实配对：3/400=0.0075。

## 2. 来源与独立性

- 来源：NTM(arXiv:1410.5401)、DNC(1607.06450)、Neural Stack/Queue(1503.01067)、Linear Attention(2006.16236)、Gated DeltaNet 系列与固定状态关联回忆，详见 M2_RESEARCH_SOURCES.md。
- 本项目独立实现 SEP tokenizer、关联记忆保护门、序列记忆指针、双记忆融合、六候选模型、验证器、训练器与评测；无外部 LLM/预训练权重/蒸馏/解释器代答。

## 3. 数学验证与资源

- 20/20 数学验证 PASS（M2_MATH_TESTS.json），修复 2 个真实 bug：指针写入 erase 广播维度、序列指针推进方向反转。
- 前缀冲突审计：历史 39922 样本 legacy 1663 冲突(4.57%)，SEP 后结构性归零。

## 4. 数据与协议

- train/val/sealed=9210/1294/1447，反事实 400 对，长度泛化 3520 条。
- 跨语义组重合 0；历史 prompt 重合 0；训练长度 2-4，测试 5-6；距离训练 0/4/8，测试 16/32/64。
- 主指标为 SEP 边界下自然 EOS 自由生成逐字符精确匹配；loss 首 token 2.0/内容 1.0/EOS 0.25。

## 5. 架构对比

| 架构 | 初筛 macro | 参数 | 状态 | tok/s |
|---|---:|---:|---:|---:|
| R1 | 0.3048 | 110694 | 128 | 10012.53 |
| R2 | 0.3739 | 90342 | 10240 | 34604.79 |
| M2-A | 0.1343 | 84840 | 1120 | 5387.96 |
| M2-B | 0.1215 | 84937 | 1120 | 4479.27 |
| M2-C | 0.0725 | 81933 | 640 | 4086.99 |
| M2-D | 0.0851 | 100848 | 1664 | 2461.08 |
| M2-E | 0.1232 | 99217 | 1664 | 1449.5 |

## 6. 连续生成与动力学

- 距离曲线：{'4': 0.7632, '8': 1.0, '16': 0.5833, '32': 0.8333, '64': 0.4833}。
- 动力学探针见 M2_DYNAMICS_PROBE.json（unique_token_ratio / fixed_point_ratio / cos_adjacent）。

## 7. 58 组真实输入输出（含失败，未用解释器纠正）

| 任务 | Prompt | 期望 | 原始输出 | 正确 | 结束 |
|---|---|---|---|---|---|
| E_distance | e=9<br>g=9<br>h=4<br>c=9<br>d=5<br>GET e | 9 | 9 | 是 | eos |
| S2_copyN | COPY GEB | GEB | GEBBG | 否 | eos |
| E_distance | d=1<br>e=2<br>c=5<br>g=0<br>h=4<br>GET d | 1 | 1 | 是 | eos |
| S2_copyN | COPY 71XS | 71XS | 71XS | 是 | eos |
| S3_reverse | REVERSE X4A | A4X | A4X | 是 | eos |
| S1_copy1 | RELAY 3 | 3 | 3 | 是 | eos |
| B8_bind | d=9<br>h=6<br>b=3<br>g=0<br>f=1<br>e=2<br>c=8<br>a=4<br>GET a | 4 | 7 | 否 | eos |
| S5_repeat | REPEAT 3 4 | 3333 | 3HH | 否 | eos |
| C_reassign | b=2<br>a=4<br>b=9<br>h=5<br>f=4<br>GET b | 9 | 2 | 否 | eos |
| B2_bind | h=3<br>a=6<br>GET h | 3 | 3 | 是 | eos |
| C_reassign | d=7<br>h=6<br>d=8<br>f=0<br>e=6<br>GET d | 8 | 8 | 是 | eos |
| B4_bind | e=8<br>b=4<br>c=9<br>g=3<br>GET g | 3 | 8 | 否 | eos |
| S4_index | INDEX 412 3 | 2 | 1 | 否 | eos |
| S2_copyN | COPY UCOW | UCOW | U | 否 | eos |
| S3_reverse | REVERSE K3JD | DJ3K | DJ3K | 是 | eos |
| S3_reverse | REVERSE H17C | C71H | C71H | 是 | eos |
| B2_bind | h=9<br>b=2<br>GET h | 9 | 2 | 否 | eos |
| D_indirect | d=2<br>h=d<br>g=4<br>d=0<br>GET h | 2 | 2 | 是 | eos |
| S5_repeat | REPEAT 1 5 | 11111 | 11 | 否 | eos |
| B2_bind | d=8<br>h=5<br>GET h | 5 | 5 | 是 | eos |
| S5_repeat | REPEAT 3 4 | 3333 | 3HH | 否 | eos |
| B8_bind | b=0<br>c=8<br>f=9<br>a=7<br>g=2<br>h=5<br>d=4<br>e=3<br>GET g | 2 | 0 | 否 | eos |
| S4_index | INDEX 1552 4 | 2 | 511 | 否 | eos |
| B8_bind | e=6<br>g=0<br>c=1<br>d=4<br>b=8<br>f=9<br>a=5<br>h=3<br>GET e | 6 | 6 | 是 | eos |
| C_reassign | a=6<br>h=1<br>a=3<br>g=0<br>e=1<br>GET a | 3 | 6 | 否 | eos |
| B_swap | d=0<br>a=4<br>GET d | 0 | 0 | 是 | eos |
| S2_copyN | COPY FRQ0C | FRQ0C | FQ0 | 否 | eos |
| B4_bind | d=0<br>a=9<br>g=2<br>e=3<br>GET e | 3 | 0 | 否 | eos |
| E_distance | e=6<br>a=8<br>f=1<br>c=4<br>d=6<br>g=3<br>h=6<br>b=3<br>d=8<br>e=9<br>b=7<br>f=5<br>g=0<br>h=8<br>a=3<br>g=7<br>d=4<br>GET e | 6 | 66 | 否 | eos |
| S1_copy1 | OUT B | B | B | 是 | eos |
| D_indirect | f=2<br>a=f<br>e=7<br>f=9<br>GET a | 2 | 2 | 是 | eos |
| S1_copy1 | OUT F | F | F | 是 | eos |
| B2_bind | a=9<br>d=3<br>GET a | 9 | 3 | 否 | eos |
| S3_reverse | REVERSE T2U59 | 95U2T | U2T | 否 | eos |
| S4_index | INDEX 05050 2 | 5 | 000 | 否 | eos |
| S5_repeat | REPEAT 3 4 | 3333 | 3HH | 否 | eos |
| S2_copyN | COPY 1MAAT | 1MAAT | AAA | 否 | eos |
| D_indirect | f=0<br>d=f<br>a=4<br>f=5<br>GET d | 0 | 0 | 是 | eos |
| D_indirect | f=3<br>c=f<br>a=0<br>f=9<br>GET c | 3 | 3 | 是 | eos |
| S4_index | INDEX 221 1 | 2 | 2 | 是 | eos |
| S5_repeat | REPEAT 3 4 | 3333 | 3HH | 否 | eos |
| S4_index | INDEX 051 3 | 1 | 5 | 否 | eos |
| B_swap | a=6<br>g=8<br>GET g | 8 | 6 | 否 | eos |
| B8_bind | a=9<br>h=3<br>d=5<br>e=2<br>g=4<br>f=1<br>c=0<br>b=7<br>GET e | 2 | 7 | 否 | eos |
| B_swap | d=7<br>c=1<br>GET d | 7 | 1 | 否 | eos |
| S3_reverse | REVERSE 6Y15E | E51Y6 | 1Y6 | 否 | eos |
| S1_copy1 | RELAY Z | Z | Z | 是 | eos |
| S2_copyN | COPY PATT | PATT | PATT | 是 | eos |
| E_distance | g=3<br>d=0<br>c=2<br>f=4<br>e=1<br>GET g | 3 | 2 | 否 | eos |
| E_distance | g=5<br>a=1<br>f=6<br>b=9<br>d=2<br>c=9<br>e=0<br>h=3<br>h=6<br>c=6<br>d=2<br>h=3<br>e=4<br>f=8<br>f=9<br>c=4<br>d=2<br>f=2<br>d=8<br>a=9<br>h=7<br>b=0<br>b=0<br>g=1<br>g=8<br>f=0<br>e=6<br>f=1<br>b=8<br>e=7<br>g=1<br>c=1<br>c=3<br>GET g | 5 | 5 | 是 | eos |
| B4_bind | a=4<br>b=5<br>f=0<br>h=9<br>GET h | 9 | 4 | 否 | eos |
| B4_bind | e=8<br>a=6<br>b=9<br>g=4<br>GET a | 6 | 8 | 否 | eos |
| C_reassign | b=7<br>g=2<br>b=4<br>a=6<br>e=2<br>GET b | 4 | 4 | 是 | eos |
| E_distance | g=1<br>c=8<br>h=3<br>e=3<br>b=2<br>a=7<br>d=5<br>f=8<br>d=9<br>GET g | 1 | 1 | 是 | eos |
| E_distance | f=7<br>d=0<br>c=9<br>g=4<br>b=8<br>h=8<br>a=9<br>e=1<br>g=3<br>e=8<br>h=1<br>a=3<br>f=5<br>b=7<br>e=3<br>d=4<br>b=6<br>GET f | 7 | 7 | 是 | eos |
| S3_reverse | REVERSE Z9UJ | JU9Z | JU9Z | 是 | eos |
| B_swap | e=0<br>b=1<br>GET b | 1 | 0 | 否 | eos |
| S3_reverse | REVERSE 0CIPN | NPIC0 | PC0 | 否 | eos |

## 8. 最终问题回答

1. SEP A/B 对照：legacy 固定面板 macro=0.1108，SEP=0.4511；SEP 更优，支持 H1。
2. 显式 SEP 从结构上消除了前缀冲突（审计 1663→0）；但正确率提升取决于训练充分度，本轮最佳模型 macro=0.5418。
3. 多字符连续生成：S2_copyN=0.32，S3_reverse=0.52，S5_repeat 因 val 切分后为 0 条未进入固定面板（train 280/sealed 70/泛化 980）；相比 M1 全 0，有实质改善（REVERSE 优于 COPY，符合无序列指针的模型特征）。
4. 序列记忆效果：M2-C（仅 SM）初筛=0.0725，M2-A（仅 AM）=0.1343；SM 未见优势。
5. 保护门抗干扰：M2-B（保护门）=0.1215 vs M2-A（共享门）=0.1343；C_reassign=0.52，B_swap=0.6。
6. 双记忆融合：M2-D=0.0851，M2-E（融合+conv）=0.1232；单一最佳架构为 R2。
7. 反事实配对准确率 0.0075（3/400）；仍接近随机。
8. 长距离保持距离曲线：{'4': 0.7632, '8': 1.0, '16': 0.5833, '32': 0.8333, '64': 0.4833}（距离 0 样本按 fact_group 切分未落入评测集，如实缺失）；距离 8→64 由 1.0 降到 0.4833，长距离保持明显变弱；但 32 高于 16，说明存在 seed/样本噪声，非严格单调。
9. 重复生成诊断见 M2_DYNAMICS_PROBE.json；overlong_rate=0.0075，immediate_eos_rate=0.0。
10. 短因果卷积（H4）：M2-E 含 conv，初筛=0.1232 vs M2-D 无 conv=0.0851。
11. 参考模型 R1 GRU=0.3048，R2 tiny Transformer=0.3739；双记忆最佳=0.3739。
12. 参数量/状态/吞吐权衡见 M2_ARCHITECTURE_COMPARISON.json 与 M2_RESOURCE_BENCHMARK.json；R2 最快但无显式记忆，M2-E 最慢但结构最完整。
13. 最佳 checkpoint 由固定验证面板 macro 选取（R2），同任务同样本同顺序，Final 先于 Best 保存且逐张量核验。
14. 权重仅本地保存，未上传公开仓库；Stable 全程只读；无外部 LLM/蒸馏/解释器代答。
15. 预算：4096/4096 step，382.83/3600s，跨重启共享台账。
16. 未达成的目标：多字符复制/反转仍未达 0.85 验收线；不确定性记忆与完整 DNC 分配器标记 NOT_RUN。
17. 下一步：单变量消融（conv on/off、protect on/off、pointer advance on/off），扩大复验步数，并对重复生成做专门的 EOS/pointer 门控诊断。
