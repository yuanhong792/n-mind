"""M3 模型共享基类。"""
from torch import nn

PAD = 0


class M2Base(nn.Module):
    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def state_elements(self):
        return 0

    def config_dict(self):
        return {"model_type": self.__class__.__name__, **getattr(self, "cfg", {})}
