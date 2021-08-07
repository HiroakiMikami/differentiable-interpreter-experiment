import torch
from app.nn._generator import FunctionGenerator
from app.nn._normalize import Normalize
from pytorch_pfn_extras.reporting import report


class Functions:
    def __init__(self, g: FunctionGenerator) -> None:
        self._values = {}
        for i in range(g.n_function):
            self._values[i] = torch.rand(g.z_dim)
        self._g = g
        self._normalize = Normalize()
        self._normalize = torch.nn.Identity()
        # self._normalize = torch.nn.LayerNorm([g.z_dim], elementwise_affine=False)

    def __getitem__(self, v: int) -> torch.Tensor:
        return self._normalize(self._values[v][None])[0]

    def optimize(self, step: int, lr: float = 1e-3) -> None:
        for z in self._values.values():
            z.requires_grad_(True)

        device = list(self._g.parameters())[0].device

        loss_fn = torch.nn.NLLLoss(reduction="sum")
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
            loss = loss_fn(pred, v[:, 0])
            loss.backward()
            optimizer.step()
        report({"functions/optimize/loss": loss / len(self._values)})

        for z in self._values.values():
            z.requires_grad_(False)
