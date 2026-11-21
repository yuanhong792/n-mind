"""router_ste 开关的梯度通路与前向等价测试（M2 收尾 F10 的回归护栏）。

钉死三件事：
  1. router_ste=False（历史行为）：稀疏前向 + 纯 action CE 反传后，router 参数无梯度
     —— 这是 gradient_diagnostics.py 实测并写进 M2 报告的"任务盲路由"事实。
  2. router_ste=True：同样的前向与反传，router 参数拿到非零梯度。
  3. router_ste=True 不改变前向数值：与 False 的输出逐位一致（STE 只改反向）。
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nmind.config import default_config
from nmind.core.architecture import NModel


def _tiny_model(router_ste: bool) -> NModel:
    cfg = default_config()
    cfg = cfg.model_copy(update={"model": cfg.model.model_copy(update={
        "z_dim": 16, "block_hidden": 16, "workspace_dim": 16, "workspace_slots": 4,
        "n_blocks": 4, "top_k": 2, "encoder_hidden": 16, "predictor_hidden": 16,
        "event_text_dim": 8})})
    torch.manual_seed(7)
    return NModel(cfg.model, n_actions=4, mem_dim=8, execution="sparse",
                  use_router=True, router_ste=router_ste)


def _action_ce_grad_norm(model: NModel, obs: torch.Tensor, acts: torch.Tensor) -> float:
    out = model.forward_sequence(obs, acts, depth=1)
    loss = torch.nn.functional.cross_entropy(
        out["action_logits"].reshape(-1, out["action_logits"].shape[-1]),
        acts.reshape(-1))
    model.zero_grad(set_to_none=True)
    loss.backward()
    total = 0.0
    for p in model.router.parameters():
        if p.grad is not None:
            total += float(p.grad.pow(2).sum())
    return total ** 0.5


def test_router_ste_gives_router_task_gradient():
    model = _tiny_model(router_ste=True)
    obs = torch.randn(2, 6, cfg_dim := model.cfg.event_numeric_dim)
    acts = torch.zeros(2, 6, dtype=torch.long)
    assert _action_ce_grad_norm(model, obs, acts) > 0.0, \
        "router_ste=True 时 action CE 必须给 router 非零梯度"


def test_router_default_stays_task_blind():
    model = _tiny_model(router_ste=False)
    obs = torch.randn(2, 6, model.cfg.event_numeric_dim)
    acts = torch.zeros(2, 6, dtype=torch.long)
    assert _action_ce_grad_norm(model, obs, acts) == 0.0, \
        "router_ste=False（历史行为）：action CE 不得给 router 任何梯度"


def test_router_ste_keeps_forward_identical():
    m_off, m_on = _tiny_model(False), _tiny_model(True)
    m_on.load_state_dict(m_off.state_dict())          # 同权重
    m_off.eval(), m_on.eval()
    obs = torch.randn(3, 5, m_off.cfg.event_numeric_dim)
    acts = torch.zeros(3, 5, dtype=torch.long)
    with torch.no_grad():
        a = m_off.forward_sequence(obs, acts, depth=1)
        b = m_on.forward_sequence(obs, acts, depth=1)
    # STE 只改反向；前向数值容许浮点舍入级差异（<1e-6）
    assert torch.allclose(a["z_hat"], b["z_hat"], atol=1e-6), "STE 不得实质改变前向数值"
    assert torch.allclose(a["action_logits"], b["action_logits"], atol=1e-6), "STE 不得改变决策输出"
