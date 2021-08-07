from app import Interpreter
import torch
from tqdm import trange


def test_interpreter() -> None:
    interpreter = Interpreter(1, 16, 16, 2, [1, 2])
    z_func = torch.rand(1, 1)
    z_args = [torch.rand(1, 1), torch.rand(1, 1), torch.rand(1, 1)]
    z_out = interpreter(z_func, z_args)
    assert len(list(interpreter.parameters())) == 16
    assert z_out.shape == (1, 1)


def test_simple_train() -> None:
    # func[0] = id, func[1] = add
    torch.manual_seed(0)
    interpreter = Interpreter(8, 16, 16, 2, [1, 2])

    interpreter.optimize_constants(100)
    optimizer = torch.optim.Adam(interpreter.parameters(), lr=1e-3)
    for _ in trange(1000, desc="training"):
        interpreter.optimize_constants(10)
        f = torch.randint(0, 2, size=()).item()
        args = list(torch.randint(-2, 3, size=(2,)))
        arg0 = args[0].item()
        arg1 = args[1].item()

        if f == 0:
            out = arg0
        else:
            out = arg0 + arg1

        # use specific values to make training easier
        z_f = interpreter.functions[f]
        z_arg0 = interpreter.values[arg0]
        z_arg1 = interpreter.values[arg1]

        optimizer.zero_grad()
        z_out = interpreter(z_f[None], [z_arg0[None], z_arg1[None]])
        pred = interpreter.value_generator(z_out)[0]
        loss = torch.nn.L1Loss()(pred, torch.tensor(out))
        loss.backward()
        optimizer.step()

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
