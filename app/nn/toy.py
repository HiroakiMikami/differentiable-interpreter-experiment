from typing import List, Union

import torch
from pytorch_pfn_extras.reporting import report


class Decoder(torch.nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.out = torch.nn.Linear(C, 3)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: [N, C]
        out = self.out(f)  # [N, 3]
        return out


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_with_logits = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.l1 = torch.nn.L1Loss(reduction="none")

    def forward(
        self, out: torch.Tensor,
        expected: List[Union[bool, int]],
    ) -> torch.Tensor:
        # out: [N, 3]
        logit_bool, logit_true, number = torch.split(out, 1, dim=1)  # list of [N, 1]
        logit_bool = logit_bool.reshape(-1)
        logit_true = logit_true.reshape(-1)
        number = number.reshape(-1)
        gt_type = torch.tensor(
            [isinstance(x, bool) for x in expected]
        ).float().to(out.device)
        gt_true = torch.tensor([int(x) for x in expected]).bool().float().to(out.device)
        gt_number = torch.tensor([int(x) for x in expected]).float().to(out.device)

        loss_type = self.bce_with_logits(input=logit_bool, target=gt_type)  # [N]
        loss_bool = self.bce_with_logits(input=logit_true, target=gt_true) * gt_type
        loss_number = self.l1(input=number, target=gt_number) * (1 - gt_type)
        report({
            "loss_type": loss_type.mean(),
            "loss_bool": loss_bool.mean(),
            "loss_number": loss_number.mean()
        })
        loss = loss_type + loss_bool + loss_number
        return loss
