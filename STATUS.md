# N-MIND 当前状态

截至 2026-11-24，项目仍是随机初始化、CPU 优先的研究原型。stable checkpoint、历史结果和历史封存集只读；没有自动晋升。

M5.1 stage0、M5.2 S0/S1 的历史数字已保留，但不解释为绑定能力：S0 可由最近实体捷径解出，S1 的输入组合与训练流重叠。stable 反事实跟随率最新为 0.130（26/200，Wilson 95% [0.090,0.184]）；旧 0.25 为 n=24 小样本估计。阶段 2.1 修复后的大组合任务通过结构门槛，但三种正控制在 100 步内均未通过，因此 N-MIND 冒烟未执行。

证据与限制见 [reports/phase21_task_fix_control.md](./reports/phase21_task_fix_control.md)。
