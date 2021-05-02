import torch
from torchnlp.encoders import LabelEncoder


class Decoder(torch.nn.Module):
    def __init__(self, C: int, value_encoder: LabelEncoder):
        super().__init__()
        n_token = value_encoder.vocab_size
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
