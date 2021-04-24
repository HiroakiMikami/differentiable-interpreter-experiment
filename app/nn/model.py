import torch


class Model(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, *args, **kwargs):
        out = self.encoder(*args, **kwargs)
        return self.decoder(out)
