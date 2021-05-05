import torch
from pytorch_pfn_extras.reporting import report


class Decoder(torch.nn.Module):
    def __init__(self, C: int):
        super().__init__()
        self.hidden = torch.nn.Linear(C, C * 4)
        self.out = torch.nn.Linear(C * 4, 3)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        # f: [N, C]
        h = self.hidden(f)  # [N, 4C]
        h = torch.tanh(h)
        out = self.out(h)  # [N, 3]
        return out


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_with_logit = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.mse_loss = torch.nn.MSELoss(reduction="none")

    def forward(
        self, out: torch.Tensor,
        expected: torch.Tensor,
    ) -> torch.Tensor:
        # out: [N, 3]
        # expected: [N, 3]
        type_loss = self.bce_with_logit(out[:, 0], expected[:, 0])
        is_bool = expected[:, 0] == 1
        bool_loss = self.bce_with_logit(out[:, 1], expected[:, 1])
        bool_loss = torch.where(is_bool, bool_loss, torch.zeros_like(bool_loss))
        num_loss = self.mse_loss(out[:, 2], expected[:, 2])
        num_loss = torch.where(is_bool, torch.zeros_like(num_loss), num_loss)

        loss = type_loss + bool_loss + num_loss
        report({
            "loss": loss.sum().detach(),
            "type_loss": type_loss.sum().detach(),
            "bool_loss": bool_loss.sum().detach(),
            "num_loss": num_loss.sum().detach(),
        })
        return loss
