import torch


class Normalize(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = x.shape[-1]
        if c == 1:
            return x
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return (x - mu) / std
