import torch
from app.nn import normalize


def test_normalize() -> None:
    torch.manual_seed(0)
    assert torch.all(normalize(torch.ones(2, 1)) == 1)
    x = torch.rand(1, 16)
    normalized = normalize(x)
    assert torch.allclose(normalized.mean(), torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(normalized.var(), torch.tensor(1.0))
    assert torch.allclose(normalize(normalize(x)), normalize(x))
