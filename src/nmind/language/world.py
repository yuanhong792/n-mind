"""Deterministic toy world with an oracle independent of the neural model."""
from __future__ import annotations
from dataclasses import dataclass
from copy import deepcopy

@dataclass
class ObjectState:
    name: str; color: str; location: str; open: bool = False

class LanguageWorld:
    def __init__(self): self.objects = {"苹果": ObjectState("苹果", "红色", "桌子"), "杯子": ObjectState("杯子", "蓝色", "桌子"), "书": ObjectState("书", "绿色", "盒子")}
    def observe(self) -> dict: return {k: vars(v).copy() for k, v in self.objects.items()}
    def reset(self): self.__init__(); return self.observe()
    def act(self, action: str, obj: str, target: str | None = None) -> dict:
        if action == "move" and obj in self.objects and target: self.objects[obj].location = target; return {"ok": True, "state": self.observe()}
        if action == "open" and obj in self.objects: self.objects[obj].open = True; return {"ok": True, "state": self.observe()}
        return {"ok": False, "state": self.observe(), "reason": "invalid action"}
    def oracle(self, question: str) -> dict:
        for name, obj in self.objects.items():
            if name in question and ("哪里" in question or "位置" in question): return {"answer": obj.location, "verified": True}
            if name in question and "什么颜色" in question: return {"answer": obj.color, "verified": True}
        return {"answer": None, "verified": False, "reason": "unknown object or question"}
