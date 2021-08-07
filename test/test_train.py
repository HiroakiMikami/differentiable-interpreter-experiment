from app import Interpreter, train
import tempfile
import torch


def test_simple_train() -> None:
    # func[0] = id, func[1] = add
    torch.manual_seed(0)
    interpreter = Interpreter(8, 16, 16, 2, [1, 2])
    with tempfile.TemporaryDirectory() as tmpdir:
        train(
            interpreter,
            [lambda x: x[0], lambda x: x[0] + x[1]],
            1,
            1000,
            1e-3,
            tmpdir,
        )

    # test id
    z_id = interpreter.functions[0]
    z_1 = interpreter.values[1]
    z_out = interpreter(z_id[None], [z_1[None], z_1[None]])
    out = interpreter.value_generator(z_out)[0]
    assert torch.abs(out - 1) < 0.5, f"actual={out.item()}, expected=id(1)=1"

    # test add
    z_add = interpreter.functions[1]
    z_m1 = interpreter.values[-1]
    z_1 = interpreter.values[1]
    z_out = interpreter(z_add[None], [z_m1[None], z_1[None]])
    out = interpreter.value_generator(z_out)[0]
    assert torch.abs(out - 0) < 0.5, f"actual={out.item()}, expected=add(-1, 1)=0"
