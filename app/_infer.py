import torch
from typing import List, NamedTuple
from app._interpreter import Interpreter
from app.nn._normalize import normalize


class Program(NamedTuple):
    z_fs: List[torch.Tensor]
    logit_args: List[torch.Tensor]

    def requires_grad_(self, x: bool) -> None:
        for z_f in self.z_fs:
            z_f.requires_grad_(x)
        for logit_arg in self.logit_args:
            logit_arg.requires_grad_(x)


def create_program(length: int, n_input: int, arity: int, z_dim: int) -> Program:
    z_fs = []
    logit_args = []
    n_var = n_input
    for i in range(length):
        z_fs.append(torch.normal(
            mean=torch.zeros(1, z_dim),
            std=torch.ones(1, z_dim)
        ))
        logit_args.append(
            torch.normal(
                mean=torch.zeros(arity, n_var),
                std=torch.zeros(arity, n_var),
            )
        )
        n_var += 1
    return Program(z_fs, logit_args)


def execute(
    program: Program,
    inputs: List[int],
    interpreter: Interpreter,
) -> torch.Tensor:
    z_inputs = [
        interpreter.value_extractor.encode(torch.tensor([[float(i)]]))
        for i in inputs
    ]

    env = torch.stack(z_inputs)  # (n_var, 1, z_dim)

    for i, z_f in enumerate(program.z_fs):
        logit_arg = program.logit_args[i]  # (n_arity, n_var)
        assert logit_arg.shape[1] == env.shape[0]
        prob_arg = torch.softmax(logit_arg, dim=1)  # (arity, n_var)

        z_args = env[:, ...] * prob_arg[:, :, None, None]  # (arity, n_var, 1, z_dim)
        z_args = z_args.sum(dim=1)  # (arity, 1, z_dim)
        z_args = list(z_args)  # list of (1, z_dim)

        z_out = interpreter.function_impl(normalize(z_f), z_args)
        env = torch.cat([env, z_out[None]], dim=0)  # (n_var + 1, 1, z_dim)
    return interpreter.value_extractor.decode(z_out)
