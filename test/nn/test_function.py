import torch
from app.nn import Function


def test_function() -> None:
    f = Function(2, 5, 3)
    assert len(list(f.parameters())) == 4

    z = f(torch.rand(1, 2), [torch.rand(1, 2), torch.rand(1, 2), torch.rand(1, 2)])
    assert z.shape == (1, 2)
