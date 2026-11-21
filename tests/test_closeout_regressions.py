"""checkpoint 重载与 detach 语义的回归护栏（M2 收尾新增）。"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nmind.config import default_config
from nmind.learning.checkpoint import load_checkpoint, save_checkpoint
from nmind.learning.losses import prediction_loss
from nmind.learning.trainer import build_model


def test_checkpoint_roundtrip_is_bitwise():
    """训练 1 步后保存→重载：同输入必须逐位一致（M2 验收门槛之一）。"""
    cfg = default_config()
    torch.manual_seed(5)
    model = build_model(cfg, seed=5, device="cpu")
    model.eval()        # 评测/保存前必须 eval：train 态前向会更新负载偏置，污染对照
    probe = torch.nn.Sequential(torch.nn.Linear(model.cfg.block_hidden, 16))
    obs = torch.randn(2, 6, cfg.model.event_numeric_dim)
    acts = torch.zeros(2, 6, dtype=torch.long)
    with torch.no_grad():
        before = model.forward_sequence(obs, acts)["action_logits"].clone()
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "t.ckpt.pt"
        save_checkpoint(model, cfg=cfg, path=p, probe=probe, meta={"seed": 5})
        loaded, _meta = load_checkpoint(p, cfg=cfg, device="cpu")
        with torch.no_grad():
            after = loaded.forward_sequence(obs, acts)["action_logits"]
    assert torch.equal(before, after), "重载后的前向输出必须逐位一致"


def test_prediction_detach_cuts_target_gradient():
    """detach_pred_target=True 时，pred 损失对 z（目标侧）的梯度必须为零。"""
    z = torch.randn(2, 6, 8, requires_grad=True)
    z_hat = torch.randn(2, 6, 8, requires_grad=True)
    prediction_loss(z_hat, z, detach_target=True).backward()
    assert z.grad is None or float(z.grad.abs().sum()) == 0.0, \
        "detach_target=True 时目标侧不得有梯度"
    assert z_hat.grad is not None and float(z_hat.grad.abs().sum()) > 0.0, \
        "预测头自身梯度必须保留（detach 不得误伤预测分支）"


def test_prediction_no_detach_keeps_target_gradient():
    z = torch.randn(2, 6, 8, requires_grad=True)
    z_hat = torch.randn(2, 6, 8, requires_grad=True)
    prediction_loss(z_hat, z, detach_target=False).backward()
    assert z.grad is not None and float(z.grad.abs().sum()) > 0.0, \
        "D 配置（不 detach）的目标侧梯度存在——这是与 action 梯度竞争的通路"
