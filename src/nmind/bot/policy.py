"""Bot Runtime 策略与硬预算（M5 Agent Runtime R1 任务书第 7 节）。

安全默认值与 configs/bot_runtime.yaml 镜像；**阈值由代码强制**，不由提示词强制。
权限持久化且不可被网页指令或神经模型修改。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BotPolicy:
    enabled: bool = True
    source_config: str = ""                      # 由 CLI 回填：本次策略来自哪个 yaml
    auto_start_with_os: bool = False
    idle_learning_enabled: bool = True
    user_task_priority: bool = True
    single_instance: bool = True
    network_enabled: bool = False
    auto_candidate_training: bool = False
    promotion_mode: str = "manual"                 # manual | approved
    allow_external_code_execution: bool = False
    allow_remote_model_api: bool = False
    practice_task_allowlist: tuple[str, ...] = ("arithmetic", "json_extract", "grid_rules")
    max_tasks_per_idle_cycle: int = 3
    min_cooldown_seconds: float = 60.0
    max_idle_cpu_percent: float = 50.0
    max_rss_mb: float = 2048.0
    max_disk_mb: float = 512.0
    max_network_requests_per_cycle: int = 0
    max_optimizer_steps_per_cycle: int = 0         # 默认 0：无授权不允许 optimizer.step
    max_training_wall_seconds_per_cycle: float = 0.0
    max_consecutive_failures: int = 3
    stable_checkpoint_readonly: bool = True
    # 显式授权后的候选更新预算（由单独命令/配置给出，默认关闭）
    candidate_training_authorized: bool = False
    candidate_max_steps: int = 48
    candidate_max_wall_s: float = 60.0
    candidate_arms: tuple[str, ...] = ("frozen", "memory", "neural", "negative")
    heldout_per_family: int = 8                  # 每族封存集大小（独立随机流，永不进学习流）
    lambda_replay: float = 0.15
    # ---- R2：技能级自主练习（能力匹配选择器）----
    skill_practice_enabled: bool = True          # 关掉就退回 R1 的工具练习-only
    capability_matrix_path: str = ""             # Gate-A 产物；空=results/bot_r2/baseline_capability_matrix.json
    practice_skill_allowlist: tuple[str, ...] = ()   # 空=用注册表里 trainable() 的全部技能
    idle_exploration_fraction: float = 0.2       # 有界确定性探索比例（外部文本不得修改）
    skill_tasks_per_idle_cycle: int = 8          # 每轮练几条学习流行（不含任何评测集行）
    # 触发阈值（online.scheduler.SchedulerConfig 的 Bot 侧镜像）：实验前预登记，不在跑动中调
    scheduler_verified_threshold: int = 32
    scheduler_max_updates_per_batch: int = 120

    def to_dict(self) -> dict:
        return dict(self.__dict__)

    @classmethod
    def from_yaml(cls, path) -> tuple["BotPolicy", list[str]]:
        """从 configs/bot_runtime.yaml 读策略，返回 (policy, 未识别键)。

        镜像文件；代码侧仍是权威实现。candidate 块只描述"授权后最多花多少"，
        晋升永远要 `bot approve --i-approve`（promotions 不接受配置文件里的开关）。
        """
        import yaml
        from pathlib import Path
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        bot = dict(data.get("bot") or {})
        cand = dict(data.get("candidate") or {})
        if "practice_task_allowlist" in bot:
            bot["practice_task_allowlist"] = tuple(bot["practice_task_allowlist"])
        known = {f for f in cls.__dataclass_fields__}
        kwargs = {k: v for k, v in bot.items() if k in known}
        if cand:
            if "max_steps" in cand:
                kwargs["candidate_max_steps"] = int(cand["max_steps"])
            if "max_wall_s" in cand:
                kwargs["candidate_max_wall_s"] = float(cand["max_wall_s"])
            if "lambda_replay" in cand:
                kwargs["lambda_replay"] = float(cand["lambda_replay"])
            if "arms" in cand:
                kwargs["candidate_arms"] = tuple(cand["arms"])
            kwargs["candidate_training_authorized"] = bool(
                cand.get("authorized") and kwargs.get("candidate_max_steps"))
        unknown = sorted(set(bot) - known)
        return cls(**kwargs), unknown


PRACTICE_FAMILIES = {
    "arithmetic": {
        "tool": "calculator",
        "render": "请计算：{a}加{b}等于多少？",
    },
    "json_extract": {
        "tool": "json_extractor",
        "render": "JSON字段提取：{payload} 中字段 {field} 的值是多少？",
    },
    "grid_rules": {
        "tool": None,
        "render": "网格规则：{question}",
    },
}
