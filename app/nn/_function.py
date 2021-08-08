import torch
from typing import List
from app.nn._normalize import normalize


class Function(torch.nn.Module):
    def __init__(self, z_dim: int, channel: int, arity: int) -> None:
        super().__init__()
        self._arity = arity
        self.l1 = torch.nn.Linear(z_dim * (1 + arity), channel)
        self.l2 = torch.nn.Linear(channel, z_dim)
        self.tanh = torch.nn.Tanh()

    @property
    def arity(self) -> int:
        return self._arity

    def forward(self, z_f: torch.Tensor, z_args: List[torch.Tensor]) -> torch.Tensor:
        assert len(z_f.shape) == 2
        for z in z_args:
            assert len(z.shape) == 2

        z = torch.cat([z_f] + z_args, dim=-1)
        h = self.tanh(self.l1(z))
        y = self.tanh(self.l2(h))
        y = normalize(y)
        return y
