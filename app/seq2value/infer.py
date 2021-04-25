from typing import Callable, List, Optional

import torch
from torchnlp.encoders import LabelEncoder


def infer(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    max_token_length: int,
    token_encoder: LabelEncoder,
    encoded_inputs: torch.Tensor,
    input_masks: torch.Tensor,
    encoded_outputs: torch.Tensor,
    n_optimize: int,
    check_interval: int,
    validate: Callable[[List[str]], bool],
    lr: float = 0.1,
) -> Optional[List[str]]:
    N = encoded_inputs.shape[1]
    n_token = token_encoder.vocab_size
    token_prob = torch.full(size=(max_token_length, n_token), fill_value=1.0 / n_token)
    token_prob.requires_grad = True

    for i in range(n_optimize):
        # [max_token_lengthn_token,] -> [max_token_length, N, n_token]
        p = token_prob.reshape(max_token_length, 1, n_token).expand(
            max_token_length, N, n_token
        )

        # Calc gradient of token prob
        optimizer = torch.optim.SGD([token_prob], lr=lr)
        optimizer.zero_grad()
        out = model(p, encoded_inputs, input_masks)
        loss = loss_fn(out, encoded_outputs).sum()
        loss.backward()
        optimizer.step()

        # normalize prob
        with torch.no_grad():
            token_prob = torch.softmax(token_prob, dim=1)

        if (i + 1) % check_interval == 0:
            token = torch.argmax(token_prob, dim=1)  # [max_token_length]
            tokens = token_encoder.batch_decode(token)
            tokens = [
                token for token in tokens
                if token != token_encoder.vocab[token_encoder.unknown_index]
            ]
            if validate(tokens):
                return tokens
    return None
