import torch
from app.nn import Normalize


def test_normalize() -> None:
    n = Normalize()
    assert len(list(n.parameters())) == 0

    assert torch.all(n(torch.ones(2, 1)) == 1)
    normalized = n(torch.rand(1, 16))
    assert torch.allclose(normalized.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(normalized.var(), torch.tensor(1.0))
