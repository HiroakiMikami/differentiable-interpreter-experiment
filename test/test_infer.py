import torch
import tempfile
from app import create_program, execute, Interpreter, train_extractor, train


def test_create_program() -> None:
    program = create_program(2, 3, 5, 7)
    assert len(program.z_fs) == 2
    assert program.z_fs[0].shape == (1, 7)
    assert program.z_fs[1].shape == (1, 7)

    assert len(program.logit_args) == 2
    assert program.logit_args[0].shape == (5, 3)
    assert program.logit_args[1].shape == (5, 4)


def test_execute() -> None:
    interpreter = Interpreter(7, 16, 5)
    program = create_program(2, 3, 5, 7)
    out = execute(program, [0, 1, 2], interpreter)
    assert out.shape == (1, 1)


def test_simple_infer() -> None:
    torch.manual_seed(0)
    # 0: constant(-1)
    # 1: constant(0)
    # 2: constant(1)
    # 3: id
    # 4: add
    interpreter = Interpreter(8, 16, 2)
    functions = [
        lambda xs: xs[0],
        lambda xs: xs[0] + xs[1],
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        train_extractor(
            interpreter,
            functions,
            2,
            1,
            10000,
            1e-4,
            tmpdir,
        )
    with tempfile.TemporaryDirectory() as tmpdir:
        train(
            interpreter,
            functions,
            1,
            10000,
            1e-4,
            tmpdir,
        )

    results = []
    for _ in range(1):
        program = create_program(2, 4, 2, interpreter.z_dim)
        program.requires_grad_(True)
        # input=[-1, 0, 1, -2]
        # expected: id(add(-2, -1)) = -3
        optimizer = torch.optim.Adam(program.z_fs + program.logit_args, lr=1e-4)
        out = torch.tensor([[-2.0]])
        for _ in range(1000):
            optimizer.zero_grad()
            pred = execute(program, [-1, 0, 1, -2], interpreter)
            loss = torch.nn.L1Loss()(pred, out)
            loss.backward()
            optimizer.step()

        f0 = int(torch.round(interpreter.function_extractor.decode(program.z_fs[0])))
        f0 = min(max(0, f0), len(functions))
        f1 = int(torch.round(interpreter.function_extractor.decode(program.z_fs[1])))
        f1 = min(max(0, f1), len(functions))

        arg_idx = []
        for logit_arg in program.logit_args:
            # (arity, var)
            _, arg_id = logit_arg.max(dim=1)
            arg_idx.append(arg_id)

        env = [-1, 0, 1, -2]
        for i in range(2):
            f = [f0, f1][i]
            arg_id = arg_idx[i]
            args = [env[int(id)] for id in arg_id]
            out = functions[f](args)
            env.append(out)
        results.append(out)
    assert -2 in results
