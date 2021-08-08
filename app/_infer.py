from pytorch_pfn_extras import reporting
import torch
from typing import List, NamedTuple
from app._interpreter import Interpreter
from app.nn._normalize import normalize
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training.triggers import ManualScheduleTrigger


class DecodedProgram(NamedTuple):
    func_ids: List[int]
    arg_idx: List[List[int]]


class Program(NamedTuple):
    z_fs: List[torch.Tensor]
    logit_args: List[torch.Tensor]

    def requires_grad_(self, x: bool) -> None:
        for z_f in self.z_fs:
            z_f.requires_grad_(x)
        for logit_arg in self.logit_args:
            logit_arg.requires_grad_(x)

    def decode(self, interpreter: Interpreter, n_function: int) -> str:
        func_ids = []
        for z_f in self.z_fs:
            f = interpreter.function_extractor.decode(z_f)
            f = torch.round(f)
            f = min(max(0, f), n_function)
            func_ids.append(int(f))

        arg_idx = []
        for logit_arg in self.logit_args:
            # (arity, var)
            _, arg_id = logit_arg.max(dim=1)
            arg_idx.append(arg_id)

        return DecodedProgram(func_ids, arg_idx)


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


def infer(
    program: Program,
    interpreter: Interpreter,
    inputs: List[List[int]],
    outputs: List[int],
    step: int,
    out_dir: str,
    lr: float = 1e-3,
) -> None:
    program.requires_grad_(True)
    optimizer = torch.optim.Adam(
        program.z_fs + program.logit_args,
        lr=lr
    )
    manager = ppe.training.ExtensionsManager(
        {}, optimizer, step,
        out_dir=out_dir,
        extensions=[],
        iters_per_epoch=1,
    )
    manager.extend(
        ppe.training.extensions.FailOnNonNumber(),
        trigger=(1000, "iteration"),
    )
    manager.extend(
        ppe.training.extensions.LogReport(
            trigger=ManualScheduleTrigger(
                list(range(100, step, 100)) + [step],
                "iteration",
            ),
            filename="log.json",
        )
    )
    manager.extend(ppe.training.extensions.ProgressBar())
    manager.extend(ppe.training.extensions.PrintReport())
    manager.extend(
        ppe.training.extensions.ProfileReport(
            filename="profile.yaml",
            append=True,
            trigger=ManualScheduleTrigger(
                list(range(1000, step, 1000)) + [step],
                "iteration",
            ),
        )
    )

    while not manager.stop_trigger:
        with manager.run_iteration():
            optimizer.zero_grad()

            loss = 0
            for i, o in zip(inputs, outputs):
                o = torch.tensor([[o]]).float()
                p = execute(
                    program,
                    i,
                    interpreter,
                )
                loss += torch.nn.L1Loss()(p, o)
            loss = loss / len(inputs)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                reporting.report({
                    "loss": loss
                })
