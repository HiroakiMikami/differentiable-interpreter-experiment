import torch
from app.nn._normalize import normalize


class FeatureExtractor(torch.nn.Module):
    def __init__(self, v_dim: int, z_dim: int, C: int):
        super().__init__()
        self._encoder = Encoder(v_dim, z_dim, C)
        self._decoder = Decoder(v_dim, z_dim, C)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def encode(self, v: torch.Tensor) -> torch.Tensor:
        return self._encoder(v)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decoder(z)


class Encoder(torch.nn.Module):
    def __init__(self, v_dim: int, z_dim: int, C: int):
        super().__init__()
        self.l1 = torch.nn.Linear(v_dim, C)
        self.l2 = torch.nn.Linear(C, z_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        return normalize(self.l2(self.tanh(self.l1(v))))


class Decoder(torch.nn.Module):
    def __init__(self, v_dim: int, z_dim: int, C: int):
        super().__init__()
        self.l1 = torch.nn.Linear(z_dim, C)
        self.l2 = torch.nn.Linear(C, v_dim)
        self.tanh = torch.nn.Tanh()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.l2(self.tanh(self.l1(z)))
