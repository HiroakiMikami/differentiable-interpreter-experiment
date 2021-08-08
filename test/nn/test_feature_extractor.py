import torch
from app.nn import Encoder, Decoder


def test_encoder() -> None:
    e = Encoder(2, 3, 5)
    assert len(list(e.parameters())) == 4

    z = e(torch.rand(1, 2))
    assert z.shape == (1, 3)


def test_decoder() -> None:
    d = Decoder(2, 3, 5)
    assert len(list(d.parameters())) == 4

    v = d(torch.rand(1, 3))
    assert v.shape == (1, 2)
