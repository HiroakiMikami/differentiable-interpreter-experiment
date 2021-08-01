import torch
from app.nn import Function, CompositeFunction


def test_function() -> None:
    f = Function(2, 5, 3)
    assert len(list(f.parameters())) == 4

    z = f([torch.rand(1, 2), torch.rand(1, 2), torch.rand(1, 2)])
    assert z.shape == (1, 2)


def test_composite_function() -> None:
    f0 = Function(2, 5, 1)
    f1 = Function(2, 5, 3)

    f = CompositeFunction([f0, f1])
    assert len(list(f.parameters())) == 8

    prob = torch.tensor([[0.8, 0.2]])
    z_args = [torch.rand(1, 2), torch.rand(1, 2), torch.rand(1, 2)]
    actual = f(prob, z_args)
    assert actual.shape == (1, 2)

    expected0 = f0(z_args[:1])
    expected1 = f1(z_args[:3])
    expected = 0.8 * expected0 + 0.2 * expected1
    assert torch.allclose(expected, actual)
