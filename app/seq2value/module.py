import math

import torch


class PositionalEncoder(torch.nn.Module):
    # original code:
    # https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec#d554
    def __init__(self, C: int, n_max_length: int):
        super().__init__()
        self.C = C
        assert self.C % 2 == 0
        pe = torch.zeros(n_max_length, C)
        for pos in range(n_max_length):
            for i in range(0, C, 2):
                pe[pos, i] = \
                    math.sin(pos / (10000 ** ((2 * i) / C)))
                pe[pos, i + 1] = \
                    math.cos(pos / (10000 ** ((2 * (i + 1)) / C)))

        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.C)
        L = x.shape[0]
        x = x + self.pe[:L]
        return x


class Module(torch.nn.Module):
    def __init__(
        self, C: int, n_layer: int, n_token: int, n_max_token_length: int,
        n_max_input_length: int, value_embedding: torch.nn.Module,
    ):
        super().__init__()
        self.token_pos_enc = PositionalEncoder(C, n_max_token_length + 1)
        self.input_pos_enc = PositionalEncoder(C, n_max_input_length + 1)
        self.token_embedding = torch.nn.Embedding(n_token, C)
        self.value_embedding = value_embedding
        self.token_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=C, dim_feedforward=C * 4, nhead=1),
            num_layers=n_layer,
        )
        self.value_encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(d_model=C, dim_feedforward=C * 4, nhead=1),
            num_layers=n_layer,
        )
        self.decoder = torch.nn.TransformerDecoder(
            torch.nn.TransformerDecoderLayer(d_model=C, dim_feedforward=C * 4, nhead=1),
            num_layers=n_layer,
        )

    def forward(
        self, token: torch.Tensor, token_mask: torch.Tensor,
        input: torch.Tensor, input_mask: torch.Tensor,
    ):
        # token: [L, N]
        # token_mask: [L, N] (True if the token is valid)
        # input: [L_in, N]
        # input_mask:[L_in, N]
        token_embed = self.token_embedding(token)  # [L, N, C]
        token_embed = self.token_pos_enc(token_embed)
        feature = self.token_encoder(
            src=token_embed, src_key_padding_mask=token_mask.permute(1, 0) == 0
        )  # [L, N, C]
        feature = feature[:1]  # gather feature of <CLS> [1, N, C]

        value_embed = self.value_embedding(input)  # [L_in, N, C]
        value_embed = self.input_pos_enc(value_embed)
        value = self.value_encoder(
            src=value_embed, src_key_padding_mask=input_mask.permute(1, 0) == 0
        )  # [L_in, N, C]
        out = self.decoder(
            tgt=value,
            memory=feature,
            tgt_key_padding_mask=input_mask.permute(1, 0) == 0,
        )  # [1, N, C]
        return out[0]
