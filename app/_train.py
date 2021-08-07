import torch
from app._interpreter import Interpreter
from typing import List, Callable
import pytorch_pfn_extras as ppe
from pytorch_pfn_extras.training.triggers import ManualScheduleTrigger


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
    manager.extend(
        ppe.training.extensions.PrintReport()  # trigger=(100, "iteration"))
    )
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

    interpreter.optimize_constants(1000)
    loss_fn = torch.nn.L1Loss()
    while not manager.stop_trigger:
        with manager.run_iteration():
            interpreter.optimize_constants(10)

            with torch.no_grad():
                fs = torch.randint(0, len(functions), size=(batch_size,))
                args = torch.randint(
                    -interpreter.values.max_value,
                    interpreter.values.max_value + 1,
                    size=(interpreter.max_arity, batch_size),
                )
                out_l = []
                for i in range(batch_size):
                    out_i = functions[fs[i]]([arg[i].item() for arg in args])
                    out_l.append(out_i)
                out = torch.tensor(out_l)

                z_f = torch.stack([interpreter.functions[f.item()] for f in fs])
                z_args = []
                for arg in args:
                    z_arg = torch.stack([interpreter.values[a.item()] for a in arg])
                    z_args.append(z_arg)

            optimizer.zero_grad()
            z_pred = interpreter(z_f, z_args)
            pred = interpreter.value_generator(z_pred)
            loss = loss_fn(pred, out)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                ppe.reporting.report({"loss": loss})
