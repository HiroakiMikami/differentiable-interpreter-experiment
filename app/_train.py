from pytorch_pfn_extras import reporting
import torch
from app._interpreter import Interpreter
from app.nn._normalize import normalize
from typing import List, Callable
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training.triggers import ManualScheduleTrigger


def train_extractor(
    interpreter: Interpreter,
    functions: List[Callable[[List[int]], int]],
    max_value: int,
    batch_size: int,
    step: int,
    lr: float,
    out_dir: str
) -> None:
    optimizer = torch.optim.Adam(interpreter.parameters(), lr)
    manager = ppe.training.ExtensionsManager(
        interpreter, optimizer, step,
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
        ppe.training.extensions.snapshot(autoload=True, n_retains=1),
        trigger=ManualScheduleTrigger(
            list(range(1000, step, 1000)) + [step],
            "iteration",
        ),
    )
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

    loss_fn = torch.nn.L1Loss(reduction="none")
    while not manager.stop_trigger:
        with manager.run_iteration():
            optimizer.zero_grad()
            with torch.no_grad():
                # sample f and args
                fs = torch.randint(0, len(functions), size=(batch_size, 1))
                args = []
                for i in range(interpreter.max_arity):
                    args.append(torch.randint(-max_value,
                                max_value + 1, size=(batch_size, 1)))
                outs_l = []
                for i in range(batch_size):
                    f = fs[i]
                    args_i = [int(arg[i]) for arg in args]
                    outs_l.append(torch.tensor([functions[f](args_i)]))
                outs = torch.stack(outs_l)

            # z
            z_fs = interpreter.function_extractor.encode(fs.float())
            z_args = [
                interpreter.value_extractor.encode(arg.float())
                for arg in args
            ]
            z_outs = interpreter.value_extractor.encode(outs.float())

            # eval function
            z_p = interpreter.function_impl(z_fs, z_args)
            p = interpreter.value_extractor.decode(z_p)

            # restore value
            p_fs = interpreter.function_extractor.decode(z_fs)
            p_args = [
                interpreter.value_extractor.decode(z_arg)
                for z_arg in z_args
            ]
            p_outs = interpreter.value_extractor.decode(z_outs)

            # eval loss
            eloss = loss_fn(p, outs).mean()

            # auto encoder loss (function)
            floss = loss_fn(p_fs, fs).mean()

            # auto encoder loss (value)
            vloss = sum([
                loss_fn(p_arg, arg) for p_arg, arg in zip(p_args, args)
            ]) + loss_fn(p_outs, outs)
            vloss = vloss.sum() / (batch_size * (interpreter.max_arity + 1))

            loss = eloss + floss + vloss
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                reporting.report({
                    "loss": loss.detach(),
                    "eval/loss": eloss.detach(),
                    "function/loss": floss.detach(),
                    "value/loss": vloss.detach(),
                })


def train(
    interpreter: Interpreter,
    functions: List[Callable[[List[int]], int]],
    batch_size: int,
    step: int,
    lr: float,
    out_dir: str
) -> None:

    optimizer = torch.optim.Adam(interpreter.parameters(), lr)
    manager = ppe.training.ExtensionsManager(
        interpreter, optimizer, step,
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
        ppe.training.extensions.snapshot(autoload=True, n_retains=1),
        trigger=ManualScheduleTrigger(
            list(range(1000, step, 1000)) + [step],
            "iteration",
        ),
    )
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

    loss_fn = torch.nn.L1Loss()
    z_dim = interpreter.z_dim
    while not manager.stop_trigger:
        with manager.run_iteration():
            with torch.no_grad():
                z_fs = normalize(torch.normal(
                    mean=torch.zeros(batch_size, z_dim),
                    std=torch.ones(batch_size, z_dim),
                ))
                z_args = [
                    normalize(torch.normal(
                        mean=torch.zeros(batch_size, z_dim),
                        std=torch.ones(batch_size, z_dim),
                    ))
                    for _ in range(interpreter.max_arity)
                ]

                fs = interpreter.function_extractor.decode(z_fs)
                args = [
                    interpreter.value_extractor.decode(z_arg)
                    for z_arg in z_args
                ]
                out_l = []
                for i in range(batch_size):
                    f_i = int(torch.round(fs[i, 0]))
                    args_i = [
                        int(torch.round(arg[i, 0]))
                        for arg in args
                    ]
                    out_l.append(torch.tensor([functions[f_i](args_i)]))
                out = torch.tensor(out_l)

            optimizer.zero_grad()
            z_preds = interpreter.function_impl(z_fs, z_args)
            preds = interpreter.value_extractor.decode(z_preds)
            loss = loss_fn(preds, out)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                ppe.reporting.report({"loss": loss})
