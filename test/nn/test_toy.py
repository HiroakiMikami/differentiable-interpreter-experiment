import numpy as np
import torch

from app.nn.toy import Decoder, Loss


def test_decoder_shape():
    decoder = Decoder(128)
    assert decoder(torch.rand(5, 128)).shape == (5, 3)


def test_loss_shape():
    loss = Loss()
    out = torch.rand(5, 3)
    expected = torch.rand(5, 3)
    assert loss(out, expected).shape == (5,)


def test_loss_match():
    loss = Loss()
    out = torch.zeros(5, 3)
    # [True, False, 0, -1, 2]
    # True
    out[0, 0] = 1e10
    out[0, 1] = 1e10
    # False
    out[1, 0] = 1e10
    out[1, 1] = -1e10
    # 0
    out[2, 0] = -1e10
    out[2, 2] = 0
    # -1
    out[3, 0] = -1e10
    out[3, 2] = -1
    # 2
    out[4, 0] = -1e10
    out[4, 2] = 2

    expected = torch.zeros(5, 3)
    expected[0, 0] = 1
    expected[0, 1] = 1
    expected[1, 0] = 1
    expected[1, 1] = 0
    expected[2, 2] = 0
    expected[3, 2] = -1
    expected[4, 2] = 2
    assert np.allclose(loss(out, expected).sum(), 0)
