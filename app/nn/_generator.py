import torch


class FunctionGenerator(torch.nn.Module):
    def __init__(self, z_dim: int, channel: int, n_function: int) -> None:
        super().__init__()
        self._n_function = n_function
        self._z_dim = z_dim
        self.l1 = torch.nn.Linear(z_dim, channel)
        self.l2 = torch.nn.Linear(channel, n_function)
        self.tanh = torch.nn.Tanh()

    @property
    def n_function(self) -> int:
        return self._n_function

    @property
    def z_dim(self) -> int:
        return self._z_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert len(z.shape) == 2

        h = self.tanh(self.l1(z))
        logit = self.l2(h)
        return torch.softmax(logit, dim=-1)


class ValueGenerator(torch.nn.Module):
    def __init__(self, z_dim: int, channel: int) -> None:
        super().__init__()
        self._z_dim = z_dim
        self.l1 = torch.nn.Linear(z_dim, channel)
        self.l2 = torch.nn.Linear(channel, 1)
        self.tanh = torch.nn.Tanh()

    @property
    def z_dim(self) -> int:
        return self._z_dim

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        assert len(z.shape) == 2
        h = self.tanh(self.l1(z))
        return self.l2(h)
