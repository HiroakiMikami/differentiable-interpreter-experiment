from app import Interpreter, train, train_extractor
from app.nn import normalize
import tempfile
import torch


def test_train_extractor() -> None:
    # func[0] = id, func[1] = add
    torch.manual_seed(0)
    interpreter = Interpreter(8, 16, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        train_extractor(
            interpreter,
            [lambda x: x[0], lambda x: x[0] + x[1]],
            2,
            1,
            1000,
            1e-3,
            tmpdir,
        )

    with torch.no_grad():
        z_id = interpreter.function_extractor.encode(torch.tensor([[0.0]]))
        z_add = interpreter.function_extractor.encode(torch.tensor([[1.0]]))

        z_m1 = interpreter.value_extractor.encode(torch.tensor([[-1.0]]))
        z_0 = interpreter.value_extractor.encode(torch.tensor([[0.0]]))
        z_1 = interpreter.value_extractor.encode(torch.tensor([[1.0]]))

        # test function extractor
        assert torch.abs(interpreter.function_extractor.decode(z_id) - 0) < 0.5
        assert torch.abs(interpreter.function_extractor.decode(z_add) - 1) < 0.5

        # test value extractor
        assert torch.abs(interpreter.value_extractor.decode(z_m1) - (-1)) < 0.5
        assert torch.abs(interpreter.value_extractor.decode(z_0) - 0) < 0.5
        assert torch.abs(interpreter.value_extractor.decode(z_1) - 1) < 0.5

        # test function
        id0 = interpreter.value_extractor.decode(interpreter.function_impl(z_id, [z_0, z_1]))
        assert torch.abs(id0 - 0) < 0.5
        add0 = interpreter.value_extractor.decode(interpreter.function_impl(z_add, [z_m1, z_1]))
        assert torch.abs(add0 - 0) < 0.5


def test_simple_train() -> None:
    # func[0] = id, func[1] = add
    torch.manual_seed(0)
    interpreter = Interpreter(8, 16, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        train_extractor(
            interpreter,
            [lambda x: x[0], lambda x: x[0] + x[1]],
            2,
            1,
            1000,
            1e-3,
            tmpdir,
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        train(
            interpreter,
            [lambda x: x[0], lambda x: x[0] + x[1]],
            1,
            1000,
            1e-3,
            tmpdir,
        )

    with torch.no_grad():
        z_id = interpreter.function_extractor.encode(torch.tensor([[0.0]]))
        z_add = interpreter.function_extractor.encode(torch.tensor([[1.0]]))

        z_m1 = interpreter.value_extractor.encode(torch.tensor([[-1.0]]))
        z_1 = interpreter.value_extractor.encode(torch.tensor([[1.0]]))

        # id
        z_out = interpreter.function_impl(z_id, [z_1, z_1])
        out = interpreter.value_extractor.decode(z_out)[0, 0]
        assert torch.abs(out - 1) < 0.5, f"actual={out.item()}, expected=id(1)=1"

        # test add
        z_out = interpreter.function_impl(z_add, [z_m1, z_1])
        out = interpreter.value_extractor.decode(z_out)[0, 0]
        assert torch.abs(out - 0) < 0.5, f"actual={out.item()}, expected=add(-1, 1)=0"



def test_simple_infer() -> None:
    # func[0] = id, func[1] = add
    torch.manual_seed(0)
    interpreter = Interpreter(8, 16, 2)
    with tempfile.TemporaryDirectory() as tmpdir:
        train_extractor(
            interpreter,
            [lambda x: x[0], lambda x: x[0] + x[1]],
            2,
            1,
            1000,
            1e-3,
            tmpdir,
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        train(
            interpreter,
            [lambda x: x[0], lambda x: x[0] + x[1]],
            1,
            1000,
            1e-3,
            tmpdir,
        )
    
    z_f = normalize(torch.normal(mean=torch.zeros(1, 8), std=torch.ones(1, 8)))
    z_c = normalize(torch.normal(mean=torch.zeros(1, 8), std=torch.ones(1, 8)))
    z_f.requires_grad_(True)
    z_c.requires_grad_(True)

    optimizer = torch.optim.Adam([z_f, z_c], lr=1e-3)
    # infer f(-2, x) => -3
    with torch.no_grad():
        z_m2 = interpreter.value_extractor.encode(torch.tensor([[-2.0]]))
        out = torch.tensor([[-3.0]])
    for _ in range(10000):
        optimizer.zero_grad()
        z_pred = interpreter.function_impl(z_f, [z_m2, z_c])
        pred = interpreter.value_extractor.decode(z_pred)
        loss = torch.nn.L1Loss()(pred, out)
        loss.backward()
        optimizer.step()
    f = int(torch.round(interpreter.function_extractor.decode(z_f)))
    c = int(torch.round(interpreter.value_extractor.decode(z_c)))

    if f == 0:
        out = -2
    else:
        out = -2 + c
    assert out == -3
