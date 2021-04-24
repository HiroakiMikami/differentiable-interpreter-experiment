from typing import List, Union

import torch
from pytorch_pfn_extras.reporting import report


class Decoder(torch.nn.Module):
    def __init__(self, C: int, max_int: int):
        super().__init__()
        n_token = max_int * 2 + 3  # True, False, 0, 1, ..., max_int, -1, ..., -max_int
        self.hidden = torch.nn.Linear(C, C * 4)
        self.out = torch.nn.Linear(C * 4, n_token)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: [N, C]
        h = self.hidden(f)  # [N, 4C]
        h = torch.tanh(h)
        out = self.out(h)  # [N, n_token]
        return out


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nll_loss = torch.nn.NLLLoss(reduction="none")

    def forward(
        self, out: torch.Tensor,
        expected: torch.Tensor,
    ) -> torch.Tensor:
        # out: [N, n_token]
        out = torch.log_softmax(out, dim=1)
        return self.nll_loss(out, expected)
