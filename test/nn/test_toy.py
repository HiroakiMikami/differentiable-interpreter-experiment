import numpy as np
import torch
from torchnlp.encoders import LabelEncoder

from app.nn.toy import Decoder, Loss


def test_decoder_shape():
    decoder = Decoder(128, LabelEncoder([1]))
    assert decoder(torch.rand(5, 128)).shape == (5, 2 + 5)


def test_loss_shape():
    loss = Loss()
    out = torch.rand(5, 7)
    expected = torch.zeros(5).long()
    assert loss(out, expected).shape == (5,)


def test_loss_match():
    loss = Loss()
    out = torch.full(size=(5, 7), fill_value=-1e10)
    out[0, 0] = 1  # True
    out[1, 1] = 1  # False
    out[2, 2] = 1  # 0
    out[3, 5] = 1  # -1
    out[4, 4] = 1  # 2
    expected = torch.tensor([0, 1, 2, 5, 4]).long()
    assert np.allclose(loss(out, expected).sum(), 0)
