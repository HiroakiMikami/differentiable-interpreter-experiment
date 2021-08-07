import torch
from app.nn._generator import ValueGenerator
from pytorch_pfn_extras.reporting import report


class Values:
    def __init__(self, g: ValueGenerator, max_value: int) -> None:
        self._values = {}
        for v in range(-max_value, max_value + 1, 1):
            self._values[v] = torch.rand(g.z_dim)
        self._g = g
        self._normalize = torch.nn.Identity()
        # self._normalize = torch.nn.LayerNorm([g.z_dim], elementwise_affine=False)

    def __getitem__(self, v: int) -> torch.Tensor:
        return self._normalize(self._values[v][None])[0]

    def optimize(self, step: int, lr: float = 1e-3) -> None:
        for z in self._values.values():
            z.requires_grad_(True)

        device = list(self._g.parameters())[0].device

        loss_fn = torch.nn.L1Loss(reduction="sum")
        optimizer = torch.optim.Adam(self._values.values(), lr=lr)
        for _ in range(step):
            optimizer.zero_grad()
            z_l = []
            v_l = []
            for v, z in self._values.items():
                z_l.append(z)
                v_l.append(torch.tensor([v]))
            z = torch.stack(z_l)  # (N, z_dim)
            v = torch.stack(v_l)  # (N, 1)
            z = z.to(device)
            v = v.to(device)
            z = self._normalize(z)
            pred = self._g(z)
            loss = loss_fn(pred, v)
            loss.backward()
            optimizer.step()
        report({"values/optimize/loss": loss / len(self._values)})

        for z in self._values.values():
            z.requires_grad_(False)
