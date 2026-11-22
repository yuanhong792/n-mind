"""Deterministic toy world with an oracle independent of the neural model."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class ObjectState:
    name: str; color: str; location: str; open: bool = False

class LanguageWorld:
    def __init__(self):
        self.objects = {
            "\u82f9\u679c": ObjectState("\u82f9\u679c", "\u7ea2\u8272", "\u684c\u5b50"),
            "\u676f\u5b50": ObjectState("\u676f\u5b50", "\u84dd\u8272", "\u684c\u5b50"),
            "\u4e66": ObjectState("\u4e66", "\u7eff\u8272", "\u76d2\u5b50"),
        }
    def observe(self) -> dict: return {k: vars(v).copy() for k, v in self.objects.items()}
    def reset(self): self.__init__(); return self.observe()
    def act(self, action: str, obj: str, target: str | None = None) -> dict:
        if action == "move" and obj in self.objects and target:
            self.objects[obj].location = target; return {"ok": True, "state": self.observe()}
        if action == "open" and obj in self.objects:
            self.objects[obj].open = True; return {"ok": True, "state": self.observe()}
        return {"ok": False, "state": self.observe(), "reason": "invalid action"}
    def oracle(self, question: str) -> dict:
        for name, obj in self.objects.items():
            if name in question and ("\u54ea\u91cc" in question or "\u4f4d\u7f6e" in question):
                return {"answer": obj.location, "verified": True}
            if name in question and "\u4ec0\u4e48\u989c\u8272" in question:
                return {"answer": obj.color, "verified": True}
        return {"answer": None, "verified": False, "reason": "unknown object or question"}
