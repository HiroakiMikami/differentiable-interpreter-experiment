import torch
from app.nn import FunctionGenerator
from app import Functions


def test_functions() -> None:
    torch.manual_seed(0)
    g = FunctionGenerator(1, 2, 2)
    functions = Functions(g)
    assert len(functions._values.keys()) == 2
    assert set(functions._values.keys()) == set(range(2))
    assert functions[0].shape == (1,)


def test_optimize() -> None:
    torch.manual_seed(0)
    g = FunctionGenerator(1, 64, 2)
    functions = Functions(g)

    # optimize generator
    optimizer = torch.optim.SGD(g.parameters(), 0.1, momentum=0.9)
    for _ in range(5000):
        optimizer.zero_grad()
        functions._values.keys()
        x = torch.tensor(list(functions._values.keys())).reshape(-1, 1).float()
        z = functions._normalize(x)
        pred = g(z)
        loss = torch.nn.L1Loss()(pred, x)
        loss.backward()
        optimizer.step()

    functions.optimize(5000, lr=1e-3)
    for v in range(2):
        prob = torch.softmax(g(functions[v][None]), dim=1)[0]
        assert prob[v] > 0.5
