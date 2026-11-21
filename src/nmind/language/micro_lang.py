"""M3.1 微型语言：程序化生成、oracle 自动判分、系统性分布外切分。

设计目标（提示词 D 节）：验证 N-MIND 能否**从随机初始化**学会一套微型语言的
实体/属性、否定、数量、关系、变量绑定（it）与多步组合，并在未见组合上泛化。
**不调用任何教师模型**：答案全部由本文件的确定性 oracle 计算。

语言（字符级、ASCII、回合制）：
    世界：实体名 × 颜色 × 位置(0..7)。颜色由世界初始化固定；位置可被 move 改变。
    回合类型与答案（oracle 逐回合维护世界状态后计算）：
      assert_color : "alpha is red ."              -> yes/no（核对真实颜色）
      assert_pos   : "alpha is at 3 ."             -> yes/no
      move         : "move alpha to 5 ."           -> ok（状态改变）
      query_color  : "what color is alpha ?"       -> "red"
      query_pos    : "where is alpha ?"            -> "5"
      query_rel    : "is alpha left of beta ?"     -> yes/no（pos_a < pos_b）
      count        : "how many red ?"              -> 数字
      negate       : "alpha is not green ."        -> yes/no（真实颜色 != green）
      变量绑定     : 名词槽位以概率 p 用 "it" 指代最近提到的实体
      多步组合     : 多个回合直接相连（状态跨回合演化）

切分（系统性 OOD，训练绝不泄漏）：
    train / dev / holdout : TRAIN_COMBOS 白名单 + train 名字池；dev/holdout 用独立种子
    OOD-name              : 全新实体名（epsilon/zeta），字符级新组合
    OOD-compose           : TRAIN_COMBOS 之外的组合 + 全代词引用（force_it=True）
    OOD-length            : 串联步数超过训练最大值

每条样本 = 一个 episode 的字符序列 + 答案位置掩码（loss/评测只看答案字符）。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TRAIN_NAMES = ("alpha", "beta", "gamma")
OOD_NAMES = ("epsilon", "zeta")
COLORS = ("red", "green", "blue")
POSITIONS = tuple(range(8))
MAX_TRAIN_CHAIN = 3                    # 训练串联步数上限（OOD-length 用 4-5）

TRAIN_COMBOS = (
    ("assert_color",), ("assert_pos",), ("move",), ("query_color",),
    ("query_pos",), ("query_rel",), ("count",), ("negate",),
    ("move", "query_pos"), ("move", "query_color"), ("query_color", "count"),
    ("move", "query_pos"), ("assert_color", "query_rel"),
)
OOD_COMBOS = (
    ("negate", "count"), ("move", "query_rel"), ("assert_pos", "count"),
    ("negate", "query_pos"), ("move", "count"),
)

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"


def build_vocab() -> dict[str, int]:
    chars: set[str] = set()
    for name in TRAIN_NAMES + OOD_NAMES:
        chars |= set(name)
    for c in COLORS:
        chars |= set(c)
    # 模板词（what/where/move/how/many/is/left/of/then/not/at/to/it/yes/no/ok）+ 数字 + 标点
    chars |= set("abcdefghijklmnopqrstuvwxyz")
    chars |= set("0123456789 .,?->|\n")
    vocab = {PAD: 0, BOS: 1, EOS: 2}
    for c in sorted(chars):
        vocab.setdefault(c, len(vocab))
    return vocab


VOCAB = build_vocab()
V = len(VOCAB)


def encode(text: str) -> list[int]:
    out = []
    for c in text:
        if c not in VOCAB:
            raise ValueError(f"词表外字符 {c!r}（文本 {text[:40]!r}）—— 词表必须先覆盖生成器")
        out.append(VOCAB[c])
    return out


@dataclass
class World:
    names: tuple[str, ...]
    color: dict[str, str] = field(default_factory=dict)
    pos: dict[str, int] = field(default_factory=dict)
    last_mentioned: str | None = None

    @classmethod
    def init(cls, names: tuple[str, ...], rng: np.random.Generator) -> "World":
        w = cls(names=names)
        for n in names:
            w.color[n] = str(rng.choice(COLORS))
            w.pos[n] = int(rng.choice(POSITIONS))
        return w


def _answer_for(step: dict, world: World) -> str:
    kind = step["kind"]
    name = step["name"]                       # 已解析实体；it 在 _sample_step 就解析
    if step["ref"] != "it":                   # 显式名词才更新绑定；it 保持原绑定
        world.last_mentioned = name
    if kind == "assert_color":
        return "yes" if world.color[name] == step["color"] else "no"
    if kind == "assert_pos":
        return "yes" if world.pos[name] == step["pos"] else "no"
    if kind == "move":
        world.pos[name] = step["pos"]
        return "ok"
    if kind == "query_color":
        return world.color[name]
    if kind == "query_pos":
        return str(world.pos[name])
    if kind == "query_rel":
        return "yes" if world.pos[name] < world.pos[step["other"]] else "no"
    if kind == "count":
        return str(sum(1 for n in world.names if world.color[n] == step["color"]))
    if kind == "negate":
        return "yes" if world.color[name] != step["color"] else "no"
    raise ValueError(f"未知 step 类型 {kind}")


def _render(step: dict) -> str:
    kind = step["kind"]
    name = step["ref"]                        # 表面词（可能是 it）
    if kind == "assert_color":
        return f"{name} is {step['color']} ."
    if kind == "assert_pos":
        return f"{name} is at {step['pos']} ."
    if kind == "move":
        return f"move {name} to {step['pos']} ."
    if kind == "query_color":
        return f"what color is {name} ?"
    if kind == "query_pos":
        return f"where is {name} ?"
    if kind == "query_rel":
        return f"is {name} left of {step['other']} ?"
    if kind == "count":
        return f"how many {step['color']} ?"
    if kind == "negate":
        return f"{name} is not {step['color']} ."
    raise ValueError(kind)


def _sample_step(rng: np.random.Generator, kind: str, world: World, *,
                 force_it: bool, use_it_prob: float) -> dict:
    # force_it 的首回合没有可绑定的 it：先用显式名词建立绑定（自然语篇也是如此）
    use_it = force_it and world.last_mentioned is not None
    if not use_it and not force_it:
        use_it = (world.last_mentioned is not None
                  and rng.random() < use_it_prob)
    if use_it and world.last_mentioned is None:
        raise ValueError("没有可绑定的实体却要求用 it")
    resolved = world.last_mentioned if use_it else str(rng.choice(world.names))
    step: dict[str, Any] = {"kind": kind, "ref": "it" if use_it else resolved,
                            "name": resolved}
    if kind in ("assert_color", "negate", "count"):
        step["color"] = str(rng.choice(COLORS))
    elif kind in ("assert_pos", "move"):
        step["pos"] = int(rng.choice(POSITIONS))
    elif kind == "query_rel":
        others = [n for n in world.names if n != resolved] or list(world.names)
        step["other"] = str(rng.choice(others))
    return step


def make_episode(rng: np.random.Generator, *, combo: tuple[str, ...],
                 names: tuple[str, ...], force_it: bool = False,
                 use_it_prob: float = 0.25) -> dict[str, Any]:
    """一个 episode：按 combo 顺序生成回合，世界状态跨回合演化。"""
    if force_it and len(names) < 2:
        raise ValueError("force_it 需要至少两个实体，否则 it 无法切换指代")
    world = World.init(names, rng)
    prompt = ""
    answers: list[str] = []
    spans: list[tuple[int, int]] = []
    steps: list[dict] = []
    for kind in combo:
        step = _sample_step(rng, kind, world, force_it=force_it,
                            use_it_prob=use_it_prob)
        text = _render(step) + " -> "
        ans = _answer_for(step, world)
        start = len(prompt) + len(text)
        prompt += text + ans + "\n"
        spans.append((start, start + len(ans)))
        answers.append(ans)
        steps.append({"kind": kind, "ref": step["ref"], "name": step["name"],
                      "other": step.get("other"), "color": step.get("color"),
                      "pos": step.get("pos"), "answer": ans, "text": _render(step)})
    return {"prompt": prompt, "answers": answers, "answer_spans": spans,
            "combo": list(combo), "steps": steps, "names": list(names),
            "force_it": force_it}


def tokenize_episode(ep: dict[str, Any]) -> dict[str, Any]:
    """字符级编码 + LM 形式的答案位置掩码：位置 t 预测 t+1 的字符，
    答案字符位置 i 的预测发生在输入位置 i-1。"""
    ids = encode(ep["prompt"])
    ans_char = [0] * len(ids)
    for a, b in ep["answer_spans"]:
        for i in range(a, b):
            ans_char[i] = 1
    loss_mask = [0] * len(ids)
    for i in range(len(ids) - 1):
        if ans_char[i + 1]:
            loss_mask[i] = 1
    return {"ids": ids, "loss_mask": loss_mask, "prompt": ep["prompt"],
            "answers": ep["answers"], "answer_spans": ep["answer_spans"],
            "combo": ep["combo"], "steps": ep["steps"]}


def make_split(split: str, *, n: int, seed: int, chain_steps: int | None = None,
               names: tuple[str, ...] | None = None, force_it: bool = False,
               ood_combo_pool: tuple[tuple[str, ...], ...] | None = None) -> dict[str, Any]:
    """生成一个 split 的 n 条样本。

    - 默认：TRAIN_COMBOS 随机取一个基组合，串联 1..MAX_TRAIN_CHAIN 步
    - chain_steps：覆盖串联长度（OOD-length 用）
    - ood_combo_pool：从给定组合池取（OOD-compose 用）
    - names / force_it：实体池与代词策略（OOD-name / OOD-compose 用）
    """
    rng = np.random.default_rng(seed)
    name_pool = names if names is not None else TRAIN_NAMES
    episodes = []
    for _ in range(n):
        if ood_combo_pool is not None:
            combo = tuple(ood_combo_pool[int(rng.integers(0, len(ood_combo_pool)))])
        else:
            k = chain_steps if chain_steps is not None else int(rng.integers(1, MAX_TRAIN_CHAIN + 1))
            base = list(TRAIN_COMBOS[int(rng.integers(0, len(TRAIN_COMBOS)))])
            combo = tuple((base * ((k // len(base)) + 1))[:k]) if k > 1 else tuple(base)
        ep = make_episode(rng, combo=combo, names=name_pool,
                          force_it=force_it)
        episodes.append(tokenize_episode(ep))
    payload = {"split": split, "n": n, "seed": seed, "names": list(name_pool),
               "force_it": force_it, "chain_steps": chain_steps,
               "episode_count": len(episodes)}
    payload["manifest_hash"] = hashlib.sha256(json.dumps(
        {k: v for k, v in payload.items()}, sort_keys=True).encode()).hexdigest()[:16]
    payload["episodes"] = episodes
    return payload


def dataset_manifest() -> dict[str, Any]:
    return {"vocab_size": V,
            "train_names": list(TRAIN_NAMES), "ood_names": list(OOD_NAMES),
            "colors": list(COLORS), "positions": len(POSITIONS),
            "train_combos": [list(c) for c in TRAIN_COMBOS],
            "ood_combos": [list(c) for c in OOD_COMBOS],
            "max_train_chain": MAX_TRAIN_CHAIN}


def decode_span(ids: list[int], span: tuple[int, int]) -> str:
    inv = {v: k for k, v in VOCAB.items()}
    a, b = span
    return "".join(inv.get(int(i), "?") for i in ids[a:b])


__all__ = ["VOCAB", "V", "encode", "make_episode", "tokenize_episode", "make_split",
           "dataset_manifest", "decode_span", "TRAIN_NAMES", "OOD_NAMES", "COLORS",
           "TRAIN_COMBOS", "OOD_COMBOS", "MAX_TRAIN_CHAIN", "PAD", "BOS", "EOS"]
