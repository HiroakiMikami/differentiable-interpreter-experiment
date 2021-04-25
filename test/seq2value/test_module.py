import numpy as np
import torch

from app.seq2value.module import Module


def test_module_shape():
    module = Module(8, 1, 3, 80, 8, torch.nn.Embedding(5, 8))
    out = module(
        torch.randint(0, 1, size=(5, 3, 3)).float(),
        torch.randint(0, 5, size=(5, 3)),
        torch.randint(0, 1, size=(5, 3)).bool(),
    )
    assert out.shape == (3, 8)


def test_module_value_mask():
    torch.manual_seed(0)
    with torch.no_grad():
        module = Module(8, 1, 3, 80, 8, torch.nn.Embedding(5, 8))
        module.eval()
        token0 = torch.randint(0, 1, size=(1, 3)).float()
        token1 = torch.randint(0, 1, size=(1, 3)).float()
        input0 = torch.randint(0, 5, size=(2,))
        input1 = torch.randint(0, 5, size=(5,))
        out0 = module(
            torch.nn.utils.rnn.pad_sequence([token0, token1]),
            torch.nn.utils.rnn.pad_sequence([input0, input1]),
            torch.tensor([[True, True], [True, True], [False, True],
                          [False, True], [False, True]]),
        )
        out1 = module(
            token0.reshape(1, 1, 3),
            input0.reshape(2, 1),
            torch.tensor([[True], [True]]),
        )
    assert np.allclose(out0[0, :], out1[0, :])
