"""配置与数据契约（Pydantic）。

设计原则：
- 所有能力上限来自配置文件，而不是模型自己决定；模型不能通过自然语言绕过预算。
- 未列出的键会报错，避免拼写错误造成"以为开了其实没开"。
- 云端默认关闭；`cloud.mode` 只能是 off / ask / auto_allowed。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

CloudMode = Literal["off", "ask", "auto_allowed"]


class ModelCfg(BaseModel):
    """N-Core 形状。放大这些数字是通向 30M–60M 参数目标的正当路径，但必须重新实测。"""

    model_config = ConfigDict(extra="forbid")

    event_numeric_dim: int = 26        # = learning.datasets.OBS_DIM（网格任务观测向量）
    event_text_dim: int = 128          # 哈希袋词特征长度（不是预训练语言模型）
    z_dim: int = 64                    # 事件潜状态 z_t
    block_hidden: int = 64             # d_h
    n_blocks: int = 8                  # 神经计算块数量
    workspace_slots: int = 8           # 固定容量
    workspace_dim: int = 64            # d_w
    top_k: int = 2                     # 默认每步激活块数
    action_dim: int = 4                # 保留的动作 one-hot 宽度
    encoder_hidden: int = 64
    predictor_hidden: int = 64
    shared_trunk_blocks: int = 0       # >0 时前若干层块跨任务共享（消融用）
    pred_mode: Literal["current", "next"] = "next"
    # next = 真世界模型目标（(s_t,a_t)→s_{t+1}，损失对齐 z[t+1] detach，与 error 门控语义
    #   一致）。2026-11-21 M2.2 验证后设为默认：legacy "current" 把预测目标当输入，
    #   实测学成恒等拷贝（cos≈0.998、动作敏感性 0.025），world_model_validated=false；
    #   next 模式 validated=true 且主任务不退化（experiments/m2_2_validation/pred_eval.json）。
    #   旧 checkpoint 重载用 allow_config_mismatch=True（形状不变，仅 pred 语义不同，
    #   决策输出不受 pred 影响）。current 保留为消融开关。


class RuntimeCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    device: Literal["auto", "cpu", "cuda"] = "auto"
    torch_threads: int = 0                    # 0 = 交给 torch 默认
    max_step_ms: float = 50.0                 # 单步硬时限，超时则降 k
    max_rss_mb: float = 4096.0
    max_recurrence_depth: int = 1             # 同一事件内额外迭代次数上限
    seed: int = 20261121


class MemoryCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    db_path: str = "data/nmind.db"
    working_capacity: int = 16                # 进程内工作记忆条数
    episodic_enabled: bool = True
    semantic_enabled: bool = True
    fts_enabled: bool = True                  # FTS5 不可用时自动退回 LIKE
    retrieval_top_k: int = 5
    fact_stale_after_days: int = 180
    consolidation_batch: int = 32


class CloudCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: CloudMode = "off"
    provider: Literal["mock", "openai_compatible", "deepseek"] = "mock"
    base_url: str = ""
    model: str = ""
    api_key_env: str = "NMIND_CLOUD_API_KEY"  # 只记变量名，绝不记值
    timeout_s: float = 20.0
    max_retries: int = 2
    max_payload_chars: int = 4000
    cost_budget_units_total: float = 0.0      # 0 = 一分都不许花
    cost_estimate_per_call: float = 0.0
    require_explicit_approval: bool = True
    # 只有白名单类别才允许出网
    allowed_categories: list[str] = Field(default_factory=lambda: ["public_web_query"])
    audit_log: str = "results/cloud_audit.jsonl"

    @field_validator("mode", mode="before")
    @classmethod
    def _mode_must_be_quoted(cls, v: Any) -> Any:
        """YAML 1.1 会把裸写的 off/on/yes/no 解析成布尔值。

        这里不猜意图：布尔值一律报错并教怎么写。云端开关是安全边界，
        把 `mode: off` 误当成 `mode: False` 然后"顺手映射成关闭"，
        和把 `mode: on` 悄悄当成"允许"一样危险。
        """
        if isinstance(v, bool):
            raise ValueError("cloud.mode 必须是字符串：请写成 mode: \"off\" / \"ask\" / \"auto_allowed\""
                             "（裸写的 off/on 会被 YAML 当成布尔值）")
        return v


class LearnCfg(BaseModel):
    """训练超参与数据规模。这些数字直接决定 CPU 训练时长，改一个就要重跑基准。"""

    model_config = ConfigDict(extra="forbid")

    lr: float = 3e-3
    batch_size: int = 16
    epochs: int = 6
    traj_len: int = 24
    n_train: int = 96
    n_test: int = 32
    n_val: int = 32                    # 调参只看 val；test 只在最终评测用一次
    n_updates: int = 6                 # 每 epoch 的优化步数（0 = 按 batch 自动）
    grad_clip: float = 5.0
    pred_weight: float = 1.0
    aux_balance_weight: float = 0.1
    var_reg_weight: float = 0.05              # 防表示塌缩
    entropy_weight: float = 0.01
    probe_weight: float = 0.25                # 连续事件记忆探针（遮挡期猜目标格）
    target_std: float = 0.25
    detach_pred_target: bool = False          # True 时预测损失不塑造编码器，仅训预测器
    replay_buffer: int = 2000
    replay_ratio: float = 0.25
    adapter_trainable: bool = True
    eval_interval: int = 1
    checkpoint_dir: str = "results"


class LanguageCfg(BaseModel):
    """语言模块。默认只用确定性的哈希文本特征，**不自动下载任何权重**。"""

    model_config = ConfigDict(extra="forbid")

    source: Literal["hash", "local_adapter"] = "hash"
    adapter_path: str = ""                    # 用户自己放好的本地模型目录
    auto_download: bool = False               # 恒为 False：程序绝不自行拉取大权重
    max_text_chars: int = 512


class LogCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    level: Literal["debug", "info", "warn", "error"] = "info"
    dir: str = "logs"
    mask_secrets: bool = True


class Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = "cpu_minimal"
    model: ModelCfg = Field(default_factory=ModelCfg)
    runtime: RuntimeCfg = Field(default_factory=RuntimeCfg)
    memory: MemoryCfg = Field(default_factory=MemoryCfg)
    cloud: CloudCfg = Field(default_factory=CloudCfg)
    learning: LearnCfg = Field(default_factory=LearnCfg)
    language: LanguageCfg = Field(default_factory=LanguageCfg)
    logging: LogCfg = Field(default_factory=LogCfg)
    offline_forced: bool = False

    @property
    def cloud_mode(self) -> CloudMode:
        return self.effective_cloud_mode()

    @property
    def learn(self) -> LearnCfg:
        return self.learning

    def effective_cloud_mode(self) -> CloudMode:
        """`--offline` 是硬开关：即便配置里配好了密钥也不出网。"""
        if self.offline_forced:
            return "off"
        return self.cloud.mode

    def cloud_key_present(self) -> bool:
        return bool(os.environ.get(self.cloud.api_key_env, ""))

    def resolve(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else (REPO_ROOT / p)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def default_config() -> Config:
    return Config()


def load_config(path: str | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """读取 YAML 配置；不存在则用内置默认；未知键直接报错。

    写法宽容：`cpu_minimal` / `cpu_minimal.yaml` / `configs/cpu_minimal.yaml` / 绝对路径都能解析，
    但**解析不到就报错**——静默回退默认值会让人以为自定义配置生效了，其实没有。
    """
    data: dict[str, Any] = {}
    if path:
        p = _resolve_config_path(path)
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if overrides:
        data = _deep_merge(data, overrides)
    return Config(**data)


def _resolve_config_path(path: str) -> Path:
    raw = Path(path)
    cands = [raw] if raw.is_absolute() else [raw, CONFIG_DIR / raw, REPO_ROOT / raw]
    if not raw.suffix:
        cands += [c.with_suffix(".yaml") for c in list(cands)]
    for c in cands:
        if c.is_file():
            return c
    tried = ", ".join(str(c) for c in cands)
    raise FileNotFoundError(f"配置文件不存在（试过：{tried}）")


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out
