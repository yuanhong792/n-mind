"""N-Learn：合成数据、损失、训练循环、经验回放与权重存取。

注意：训练循环放在 trainer.py 而不是 train.py —— 模块与同名函数会在包命名空间里互相
遮蔽（from .train import train 之后 learning.train 就成了函数），改名是一次性解决。
"""

from . import datasets
from .checkpoint import (config_hash, git_revision, list_snapshots, load_checkpoint,
                         load_probe, load_snapshot, probe_state, save_checkpoint, save_snapshot)
from .losses import LossWeights, compute_total
from .replay import ReservoirReplay, mix_batch
from .trainer import TrainResult, build_model, evaluate, make_data, train, train_continual

__all__ = ["datasets", "train", "train_continual", "evaluate", "build_model", "make_data",
           "TrainResult", "LossWeights", "compute_total", "ReservoirReplay", "mix_batch",
           "save_checkpoint", "load_checkpoint", "config_hash", "git_revision",
           "save_snapshot", "load_snapshot", "list_snapshots", "probe_state", "load_probe"]
