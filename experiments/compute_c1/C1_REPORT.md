# N-MIND C1.0 计算能力优先学习 — 真实训练报告

> 分支 `experiment/compute-first-c1`｜实际 optimizer.step = **2048**｜墙钟 451.1s｜Random init SHA `953a26be18d815ca`
> 未使用外部 LLM / 预训练权重 / 蒸馏；Python 只生成标签与验证，不参与模型推理。

## 0. 环境与口径
- 项目路径：`D:\cpu本地模型开发\n-mind`（任务书路径 `.venv-gate24` 本机不存在），Python 3.12.10 + torch 2.10.0+cpu（CUDA 不可用）。
- 计算 Tokenizer：`7daa9040116a02ae`，字符级 101 token，无 BPE 合并；训练/推理同源。
- 模型：N-Cell(4块/top2选2)/Workspace(8槽)，参数量 106410；从随机初始化开始。
- 主指标为**贪心自由生成**（非 teacher forcing）；完整答案含 EOS。
- 独立泛化评测（C1_GENERALIZATION.json）：C0 超长复制/反转（length>4）准确率均为 **0%**（未获得长度外推）；C2_bindN 未见组合准确率 **0.43**（存在一定组合泛化证据）。

## 1. 能力等级（自由生成）
| 等级 | 样本 | 准确率 | 状态 |
|---|---:|---:|---|
| C0_symbol_sequence | 504 | 8.93% | TRAINED |
| C2_variable_binding | 961 | 46.72% | TRAINED |
| C3_arithmetic | 282 | 39.72% | TRAINED |
| C4_logic | 0 | — | INTERFACE_ONLY |
| C5_program_understanding | 0 | — | INTERFACE_ONLY |
| C6_program_generation | 0 | — | INTERFACE_ONLY |
| C7_error_fixing | 0 | — | INTERFACE_ONLY |
| C8_autonomous_learning | 0 | — | INTERFACE_ONLY |
| C9_autoprogram | 0 | — | INTERFACE_ONLY |

- 反事实成对正确率：**14.20%**（500 对）
- Router 稀疏空间：**已确认**，激活率 0.5，选择集合数 6；改评分改变前向 = True；Router 非零梯度 = True。
- Workspace 平均占用 1.0471，首/末 0.25/0.9449。

## 2. 分任务明细
| 任务 | n | 自由生成 | 立即EOS | 空输出 | 推理秒/样本 |
|---|---:|---:|---:|---:|---:|
| C0_copy1 | 7 | 14.29% | 14.29% | 14.29% | 0.00727 |
| C0_copyN | 156 | 0.00% | 33.97% | 33.97% | 0.00269 |
| C0_index | 166 | 26.51% | 16.87% | 16.87% | 0.00083 |
| C0_next | 2 | 0.00% | 100.00% | 100.00% | 6e-05 |
| C0_reverse | 173 | 0.00% | 100.00% | 100.00% | 6e-05 |
| C2_bind1 | 18 | 100.00% | 0.00% | 0.00% | 0.00112 |
| C2_bindN | 467 | 35.55% | 0.00% | 0.00% | 0.00106 |
| C2_counter | 275 | 40.36% | 0.00% | 0.00% | 0.00106 |
| C2_indirect | 72 | 100.00% | 0.00% | 0.00% | 0.00102 |
| C2_reassign | 129 | 63.57% | 0.00% | 0.00% | 0.00101 |
| C3_arith | 39 | 25.64% | 5.13% | 5.13% | 0.00087 |
| C3_var_arith | 243 | 41.98% | 0.00% | 0.00% | 0.00098 |

## 3. 训练曲线（已教任务 val acc）
| step | 阶段 | taught acc | loss | 墙钟 |
|---:|---|---:|---:|---:|
| 1 | C0_foundation | 0.00% | 4.63043 | 2.5s |
| 128 | C0_foundation | 0.00% | 3.24742 | 14.9s |
| 256 | C0_foundation | 0.00% | 3.11011 | 27.9s |
| 384 | C2_binding | 0.00% | 2.17678 | 54.1s |
| 512 | C2_binding | 26.04% | 1.73064 | 80.7s |
| 640 | C2_binding | 22.92% | 1.49485 | 106.3s |
| 768 | C2_binding | 33.33% | 1.37715 | 132.6s |
| 896 | C3_math | 31.25% | 1.34739 | 158.1s |
| 1024 | C3_math | 32.29% | 1.27487 | 184.3s |
| 1152 | C3_math | 38.54% | 1.23026 | 210.9s |
| 1280 | C3_math | 34.38% | 1.21189 | 239.0s |
| 1408 | C3_math | 36.46% | 1.20004 | 273.6s |
| 1536 | C3_math | 35.42% | 1.14912 | 313.0s |
| 1664 | C3_consolidate | 35.42% | 1.39788 | 353.6s |
| 1792 | C3_consolidate | 31.25% | 1.30241 | 392.5s |
| 1920 | C3_consolidate | 30.21% | 1.26135 | 426.2s |
| 2048 | C3_consolidate | 31.25% | 1.27504 | 451.0s |

## 4. 真实样本（成功与失败）
成功样例：
- `p=0
f=4
ADD p f` → 期望 `4`｜模型 `4` ✓
- `c=9
i=4
GET i` → 期望 `4`｜模型 `4` ✓
- `j=0
m=5
SUB j m` → 期望 `0`｜模型 `0` ✓
- `INDEX Q2 2` → 期望 `2`｜模型 `2` ✓
- `f=3
k=7
f=4
f=4
GET f` → 期望 `4`｜模型 `4` ✓
- `p=2
i=p
GET i` → 期望 `2`｜模型 `2` ✓
- `i=2
c=7
GET c` → 期望 `7`｜模型 `7` ✓
- `i=2
j=5
h=0
n=7
GET h` → 期望 `0`｜模型 `0` ✓
- `o=8
l=4
p=2
d=4
GET l` → 期望 `4`｜模型 `4` ✓
- `l=1
d=5
MUL l d` → 期望 `5`｜模型 `5` ✓
- `d=8
j=9
o=2
a=2
m=7
c=7
l=2
g=7
GET g` → 期望 `7`｜模型 `7` ✓
- `a=7
b=a
GET b` → 期望 `7`｜模型 `7` ✓
- `i=3
h=3
SUB i h` → 期望 `0`｜模型 `0` ✓
- `i=6
j=4
b=1
p=5
k=3
c=6
o=0
g=5
GET p` → 期望 `5`｜模型 `5` ✓
- `o=1
f=1
DIV o f` → 期望 `1`｜模型 `1` ✓
- `k=2
m=k
GET m` → 期望 `2`｜模型 `2` ✓
- `f=7
l=9
f=8
f=9
GET f` → 期望 `9`｜模型 `9` ✓
- `b=0
l=2
DIV b l` → 期望 `0`｜模型 `0` ✓
- `f=3
k=2
o=6
c=6
GET c` → 期望 `6`｜模型 `6` ✓
- `e=0
k=3
MUL e k` → 期望 `0`｜模型 `0` ✓
- `k=0
p=k
GET p` → 期望 `0`｜模型 `0` ✓
- `INDEX B5 2` → 期望 `5`｜模型 `5` ✓
失败样例：
- `INDEX SQ2P 3` → 期望 `2`｜模型 `Q` ✗
- `g=4
l=4
MUL g l` → 期望 `16`｜模型 `4` ✗
- `COPY 1XJ` → 期望 `1XJ`｜模型 `JJ` ✗
- `c=4
i=9
GET i` → 期望 `9`｜模型 `4` ✗
- `e=2
h=9
o=1
g=5
n=1
k=4
j=8
f=9
GET f` → 期望 `9`｜模型 `4` ✗
- `i=7
c=2
GET c` → 期望 `2`｜模型 `7` ✗
- `p=5
a=8
f=0
m=1
GET a` → 期望 `8`｜模型 `0` ✗
- `g=2
a=5
MUL g a` → 期望 `10`｜模型 `5` ✗
- `i=2
j=5
h=7
n=0
GET h` → 期望 `7`｜模型 `0` ✗
- `p=5
g=5
MUL p g` → 期望 `25`｜模型 `5` ✗
- `p=3
i=5
MUL p i` → 期望 `15`｜模型 `5` ✗
- `o=2
a=7
i=3
d=2
GET i` → 期望 `3`｜模型 `2` ✗
- `a=0
k=7
n=7
l=3
j=3
h=4
d=1
m=6
GET d` → 期望 `1`｜模型 `4` ✗
- `k=4
l=3
SUB k l` → 期望 `1`｜模型 `0` ✗
- `g=8
n=2
DIV g n` → 期望 `4`｜模型 `8` ✗
- `REVERSE TM` → 期望 `MT`｜模型 `` ✗
- `INDEX O9ILV 3` → 期望 `I`｜模型 `9` ✗
- `a=3
m=3
h=4
b=6
GET m` → 期望 `3`｜模型 `6` ✗
- `REVERSE EK8` → 期望 `8KE`｜模型 `` ✗
- `REVERSE 130Q` → 期望 `Q031`｜模型 `` ✗
- `COPY G9` → 期望 `G9`｜模型 `99999999` ✗
- `COPY TOD` → 期望 `TOD`｜模型 `TT` ✗
- `REVERSE B0G4D` → 期望 `D4G0B`｜模型 `` ✗

## 5. 十问回答
1. 符号复制：见 C0_copy1/C0_copyN 行，达标与否以自由生成准确率为准，未达标不宣称学会。
2. 记住多个变量：见 C2_bindN 各行；随变量数增加的表现差异在 C1_EVALUATION.json 分任务表中。
3. 按查询提取变量：C2_bind1/bindN 自由生成准确率即答案；反事实成对正确率见上。
4. 变量重赋值：见 C2_reassign 行。
5. 数学运算：见 C3_arith/C3_var_arith 行。
6. 简单程序理解：本轮为接口（MiniLang + environment），未宣称模型已理解。
7. 程序生成：仅解释器/训练/评估接口就绪，未训练、未宣称。
8. 记忆 vs 泛化：训练内组合高准确率属记忆证据；未见组合/长度泛化需对比 C0_copyN 的 length>in_distribution 与 sealed 分任务，当前若无单独泛化列则标记 NOT_YET_VERIFIED。
9. N-Cell/Router/Workspace 作用：Router 稀疏空间与负载见 C1_ROUTER_AUDIT.json；Workspace 占用见 C1_MEMORY_AUDIT.json；N-Cell 层级可解码性本轮未逐层探针，NOT_YET_VERIFIED。
10. 距离自主编程：仍缺稳定的多变量泛化、组合程序执行预测、执行反馈权重更新（仅接口）。

## 6. 诚实边界
- 本轮不因某项简单任务成功就宣称通用智能。
- C4–C9 未训练/未达标的项目均明确标注 INTERFACE_ONLY。
- 权重仅本地保存，未上传 GitHub；Stable 与历史语言实验未改动。
