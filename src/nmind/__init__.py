"""N-MIND 2.0（Neural Memory & Incremental Dynamics）。

一个刻意做小的研究用模型：目标是**在普通 x86-64 CPU 上完成
"感知事件 → 更新内部状态 → 选择计算块 → 读记忆 → 决策 → 校验 → 巩固"
的闭环，并且能用本机算力持续学习**。

三条口径必须分开，任何文档与输出都不许混用（详见 docs/LIMITATIONS.md）：
  1. 软件功能完成 ≠ 训练获得能力 ≠ 达到通用智能；
  2. 网络参数里的能力由 learning/ 度量，SQLite 里的事实由 memory/ 度量；
  3. 所有性能数字来自本机可复现实测，缺传感器就写"未测量"。

入口：
    python -m nmind.cli doctor
    python -m nmind.cli demo --epochs 20
"""

from __future__ import annotations

__version__ = "0.1.0"

# 刻意不在这里 import 子模块：`import nmind` 必须保持轻量（不加载 torch）。
# 需要时按 `from nmind.core.architecture import NModel` 这样的路径显式取。
__all__ = ["__version__"]
