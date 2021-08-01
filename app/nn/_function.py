import torch
from typing import List


class Function(torch.nn.Module):
    def __init__(self, z_dim: int, channel: int, arity: int) -> None:
        super().__init__()
        self._arity = arity
        self.l1 = torch.nn.Linear(z_dim * arity, channel)
        self.l2 = torch.nn.Linear(channel, z_dim)
        self.tanh = torch.nn.Tanh()
        self.norm = torch.nn.LayerNorm([z_dim], elementwise_affine=False)

    @property
    def arity(self) -> int:
        return self._arity

    def forward(self, z_args: List[torch.Tensor]) -> torch.Tensor:
        for z in z_args:
            assert len(z.shape) == 2

        z = torch.cat(z_args, dim=-1)
        h = self.tanh(self.l1(z))
        y = self.tanh(self.l2(h))
        return self.norm(y)


class CompositeFunction(torch.nn.Module):
    def __init__(self, functions: List[Function]) -> None:
        super().__init__()
        self.functions = torch.nn.ModuleList(functions)

    def forward(self, prob: torch.Tensor, z_arg: List[torch.Tensor]) -> torch.Tensor:
        assert len(prob.shape) == 2
        assert prob.shape[1] == len(self.functions)

        out_l = []
        for f in self.functions:
            out_l.append(f(z_arg[:f.arity]))
        out = torch.stack(out_l, dim=1)  # (N, n_func, C)
        return (prob[:, :, None] * out).sum(dim=1)
