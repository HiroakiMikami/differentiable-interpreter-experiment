import torch


class FunctionGenerator(torch.nn.Module):
    def __init__(self, z_dim: int, channel: int, n_function: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(z_dim, channel)
        self.l2 = torch.nn.Linear(channel, n_function)
        self.tanh = torch.nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert len(z.shape) == 2

        h = self.tanh(self.l1(z))
        logit = self.l2(h)
        return torch.softmax(logit, dim=-1)


class ValueGenerator(torch.nn.Module):
    def __init__(self, z_dim: int, channel: int) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(z_dim, channel)
        self.l2 = torch.nn.Linear(channel, 1)
        self.tanh = torch.nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert len(z.shape) == 2
        h = self.tanh(self.l1(z))
        return self.l2(h)
