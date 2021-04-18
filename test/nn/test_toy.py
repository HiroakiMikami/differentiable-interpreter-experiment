import numpy as np
import torch

from app.nn.toy import Decoder, Loss


def test_decoder_shape():
    decoder = Decoder(128)
    assert decoder(torch.rand(5, 128)).shape == (5, 3)


def test_loss_shape():
    loss = Loss()
    out = torch.rand(5, 3)
    expected = [True, False, 0, 1, 2]
    assert loss(out, expected).shape == (5,)


def test_loss_match():
    loss = Loss()
    out = torch.rand(5, 3)
    out[:2, 0] = 1e10  # bool
    out[2:, 0] = -1e10  # number
    out[0, 1] = 1e10
    out[1, 1] = -1e10
    out[2, 2] = 0.0
    out[3, 2] = 1.0
    out[4, 2] = 2.0
    expected = [True, False, 0, 1, 2]
    assert np.allclose(loss(out, expected).sum(), 0)
