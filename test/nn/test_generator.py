import torch
from app.nn import FunctionGenerator, ValueGenerator


def test_function_generator() -> None:
    g = FunctionGenerator(2, 3, 5)
    assert len(list(g.parameters())) == 4

    pred = g(torch.rand(1, 2))
    assert pred.shape == (1, 5)
    total = pred.sum()
    assert torch.all(0 <= pred)
    assert torch.all(pred <= 1)
    assert torch.allclose(total, torch.tensor(1.0))


def test_value_generator() -> None:
    g = ValueGenerator(2, 3)
    assert len(list(g.parameters())) == 4

    v = g(torch.rand(1, 2))
    assert v.shape == (1, 1)
