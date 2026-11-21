"""GlobalWorkspace：固定容量的共享状态槽与竞争写入。

为什么需要显式竞争：多条写入若直接相加就会互相覆盖（等价于把信息求平均后淹没）。
这里用"每个写入按与槽键的相似度做软分配 + 按占用度归一化融合"，并且保留槽效用分数。

占用度的单位（重要，别猜）：每条写入贡献"强度"那么多质量（默认强度 1），按分配矩阵摊到各槽，
所以 `occupied` 是"这个槽装了几条写入"，`capacity` 是"一个槽最多装几条"。
强度 0 就是没写：既不占位也不稀释旧内容；写入幅度同样不改变占用度。
这个单位与 d_w、嵌入尺度都无关——换配置、换维度时容量的含义不会悄悄漂移，
代价是它不表示"内容多少"，一条内容很小但强度为 1 的写入照样占一整条位置
（见 tests/test_workspace.py 的守恒与强度测试）。

容量是硬上限：一步之内每个超容量的槽都会被反复腾到上限以下，
被腾走的部分作为溢出记录返回（保留一半内容，不是清零），调用方把它写进情景记忆，
不允许上下文无限增长。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class WorkspaceRead:
    value: torch.Tensor      # [B, d_w] 读出
    attn: torch.Tensor       # [B, S]   各槽被读到的权重
    occupied: torch.Tensor   # [B, S]   各槽当前占用度


@dataclass
class WorkspaceWrite:
    state: torch.Tensor      # [B, S, d_w] 新槽内容
    occupied: torch.Tensor   # [B, S]
    utility: torch.Tensor    # [B, S]
    assignments: torch.Tensor  # [B, M, S] 每条写入实际落到各槽的量（softmax×强度）
    overflow: list[dict] = field(default_factory=list)  # 被挤出的槽，待持久化


class GlobalWorkspace(nn.Module):
    def __init__(
        self,
        z_dim: int,
        d_w: int,
        slots: int = 8,
        read_dim: int | None = None,
        capacity: float = 1.5,
        decay: float = 0.9,
    ) -> None:
        super().__init__()
        if slots < 1:
            raise ValueError("slots 必须 >= 1")
        self.d_w = d_w
        self.slots = slots
        self.capacity = float(capacity)
        self.decay = float(decay)

        self.slot_keys = nn.Parameter(torch.randn(slots, d_w) * (1.0 / d_w ** 0.5))
        self.read_dim = read_dim or d_w
        self.q_proj = nn.Linear(z_dim, self.read_dim)
        self.k_proj = nn.Linear(d_w, self.read_dim, bias=False)
        self.v_proj = nn.Linear(d_w, d_w, bias=False)   # 读出必须是"槽里的内容"：
        # 带 bias 时全空工作空间也会读出一个非零常向量，下游分不清"没东西"和"有东西"。
        self.write_proj = nn.Linear(d_w, d_w, bias=False)   # 无 bias：零强度写入必须真的"什么都没写"，
        # 带 bias 时空 payload 也会往槽里注入一个常向量，下游分不清"没写"和"写了个常量"。
        self.utility_mlp = nn.Linear(d_w, 1)

    # ------------------------------------------------------------------

    def initial_state(self, batch: int, device=None) -> dict[str, torch.Tensor]:
        return {
            "content": torch.zeros(batch, self.slots, self.d_w, device=device),
            "occupied": torch.zeros(batch, self.slots, device=device),
            "utility": torch.zeros(batch, self.slots, device=device),
        }

    def read(self, state: dict[str, torch.Tensor], query_z: torch.Tensor) -> WorkspaceRead:
        content = state["content"]
        q = self.q_proj(query_z)[:, None, :]                     # [B, 1, r]
        k = self.k_proj(content)                                 # [B, S, r]
        attn = torch.softmax(
            (q * k).sum(-1) / (self.read_dim ** 0.5) + self._mask_from_occupied(state["occupied"]),
            dim=-1,
        )
        value = torch.bmm(attn[:, None, :], self.v_proj(content)).squeeze(1)   # [B, d_w]
        return WorkspaceRead(value=value, attn=attn, occupied=state["occupied"])

    def _mask_from_occupied(self, occupied: torch.Tensor) -> torch.Tensor:
        """空槽给出很强的负偏置，避免注意力被零向量稀释；但全部为空时保持均匀。"""
        empty = (occupied <= 1e-6).float()
        all_empty = (empty.sum(-1, keepdim=True) >= self.slots).float()
        return -8.0 * empty * (1.0 - all_empty)

    def write(
        self,
        state: dict[str, torch.Tensor],
        payloads: torch.Tensor,
        strengths: torch.Tensor | None = None,
    ) -> WorkspaceWrite:
        """payloads: [B, M, d_w]；strengths: [B, M]（可选，默认 1）。

        返回值里的 `overflow` 每条形如
        `{batch, slot, content, occupied, evicted_mass}`：
        `content` 是腾出之后**留在槽里**的那份内容（保留一半以上，不是凭空失忆），
        `evicted_mass` 是被从该槽腾走的占用量，调用方把它写进情景记忆才算真正"溢出落地"。
        """
        bsz, m, dim = payloads.shape
        if dim != self.d_w:
            raise ValueError(f"payload 维度应为 {self.d_w}，实际 {dim}")
        if strengths is None:
            mass_w = torch.ones(bsz, m, device=payloads.device)
        else:
            if strengths.shape != (bsz, m):
                raise ValueError(f"strengths 形状应为 ({bsz}, {m})，实际 {tuple(strengths.shape)}")
            mass_w = strengths.clamp(0.0, 4.0)
        v = self.write_proj(payloads)                                       # [B,M,d] 内容方向

        k = self.slot_keys.unsqueeze(0).expand(bsz, -1, -1)                 # [B,S,d]
        logits = torch.bmm(v, k.transpose(1, 2)) / (self.d_w ** 0.5)        # [B,M,S]
        # 只减**异质**占用才有竞争效果：softmax 对整体平移不变，
        # 所有槽同样拥挤时这项会被抵消，分配结果与不减完全一致（见 tests/test_workspace.py）。
        logits = logits - 2.0 * state["occupied"].unsqueeze(1)
        assign = torch.softmax(logits, dim=-1)                              # [B,M,S] 落在哪些槽
        # 强度决定"落多少"，不影响"落到哪"：strength=0 就是没写（不占位、不稀释），
        # 这样占用度的单位是"写入条数×强度"，仍然与嵌入维度无关。
        alloc = assign * mass_w[:, :, None]                                 # [B,M,S]

        # 融合：新内容 = (旧内容*旧占用 + Σ 分配*写入) / (旧占用 + Σ 分配)
        incoming = torch.bmm(alloc.transpose(1, 2), v)                      # [B,S,d]
        # alloc 每行和为该写入的强度 ⇒ 槽占用求和 = Σ 强度（默认每条 1）。
        # 占用度的单位是"条目×强度"，与嵌入维度、内容幅度都无关；
        # 总占用 = decay*旧占用 + Σ强度（守恒，tests 里逐条钉住）。
        incoming_mass = alloc.sum(dim=1)                                    # [B,S]
        old = state["content"]
        old_mass = state["occupied"] * self.decay
        denom = (old_mass + incoming_mass).clamp_min(1e-6)
        new_content = (old * old_mass.unsqueeze(-1) + incoming) / denom.unsqueeze(-1)
        new_occupied = denom

        # 溢出处理：容量是硬上限，所以这里必须把**每个**超容量的槽都腾下来，
        # 而不是每个样本只腾一个——后者在多个槽同时拥挤时会留下 occupied > capacity，
        # 那句话就变成谎话了（tests/test_workspace.py 抓住了这个洞）。
        # 衰减用掩码乘法实现，绝不对带计算图的张量做 index_put_：
        # 那会破坏反向图，轻则报错、重则静默算错梯度。
        decay = torch.ones_like(new_occupied)
        overflow: list[dict] = []
        evicted = torch.zeros_like(new_occupied)
        eff_utility = torch.sigmoid(self.utility_mlp(new_content)).squeeze(-1) * new_occupied
        occ = new_occupied * decay
        # 有限轮：每轮把每个 batch 里效用最低的拥挤槽砍一半，最多 4*slots 轮必然收敛
        # （每轮至少让一个拥挤槽的占用减半），不会死循环。
        for _ in range(4 * self.slots):
            crowded = occ > self.capacity
            if not bool(crowded.any()):
                break
            for b in torch.nonzero(crowded.any(dim=-1), as_tuple=False).flatten().tolist():
                # 只在仍拥挤的槽里选，效用低的先被腾出
                cand = torch.where(crowded[b], eff_utility[b], torch.full_like(eff_utility[b], float("inf")))
                s = int(torch.argmin(cand).item())
                before = float(occ[b, s].item())
                decay[b, s] *= 0.5          # 溢出槽腾出：保留一半内容，避免完全失忆
                evicted[b, s] += before * 0.5
            occ = new_occupied * decay
        # 削峰：极端写入量下减半仍不够时，直接把剩余超容量槽缩到容量上限。
        still = occ > self.capacity
        if bool(still.any()):
            scale = (self.capacity / occ.clamp_min(1e-6)).clamp(max=1.0)
            scale = torch.where(still, scale, torch.ones_like(scale))
            evicted = evicted + occ * (1.0 - scale) * still.float()
            decay = decay * scale
            occ = new_occupied * decay
        for b, s in torch.nonzero(evicted > 0, as_tuple=False).tolist():
            overflow.append({
                "batch": int(b),
                "slot": int(s),
                "content": (new_content[b, s] * decay[b, s]).detach().cpu().numpy().tolist(),
                "occupied": float(occ[b, s].item()),
                "evicted_mass": round(float(evicted[b, s].item()), 6),
            })
        new_content = new_content * decay.unsqueeze(-1)
        new_occupied = occ
        utility = torch.sigmoid(self.utility_mlp(new_content)).squeeze(-1) * new_occupied

        new_state = {"content": new_content, "occupied": new_occupied, "utility": utility}
        return WorkspaceWrite(
            state=new_state,
            occupied=new_occupied,
            utility=utility,
            assignments=alloc,
            overflow=overflow,
        )

    def pressure(self, occupied: torch.Tensor) -> dict:
        return {
            "mean_occupied": round(float(occupied.mean().item()), 4),
            "crowded_slots": int((occupied > self.capacity).sum().item()),
            "used_slots": int((occupied > 1e-6).sum().item()),
            "capacity_slots": self.slots,
        }
