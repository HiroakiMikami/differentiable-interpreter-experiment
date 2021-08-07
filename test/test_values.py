import torch
from app.nn import ValueGenerator
from app import Values


def test_values() -> None:
    torch.manual_seed(0)
    g = ValueGenerator(1, 2)
    values = Values(g, 5)
    assert len(values._values.keys()) == 11
    assert set(values._values.keys()) == set(range(-5, 6, 1))
    assert values[0].shape == (1,)


def test_optimize() -> None:
    g = ValueGenerator(1, 64)
    values = Values(g, 2)

    # optimize generator
    optimizer = torch.optim.SGD(g.parameters(), 0.1, momentum=0.9)
    for _ in range(5000):
        optimizer.zero_grad()
        values._values.keys()
        x = torch.tensor(list(values._values.keys())).reshape(-1, 1).float()
        z = values._normalize(x)
        pred = values._g(z)
        loss = torch.nn.L1Loss()(pred, x)
        loss.backward()
        optimizer.step()

    values.optimize(5000, lr=1e-3)
    for v in range(-2, 3, 1):
        y = g(values[v][None])[0]
        assert torch.abs(y - v).item() < 1, f"actual={y.item()}, expected={v}"
