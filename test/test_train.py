from app import Interpreter, train, train_extractor
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
