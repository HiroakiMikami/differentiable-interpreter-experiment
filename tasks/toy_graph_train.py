import argparse
import logging
import os
import sys

import numpy as np
import pytorch_pfn_extras as ppe
import torch
import yaml
from pytorch_pfn_extras.training import extensions

from app.datasets.toy import (
    Boolean,
    Example,
    FlatSample,
    Function,
    Input,
    Interpreter,
    Number,
    Parser,
    Program,
    RandomFlatDataset,
)
from app.graph.module import Module
from app.nn.toy import Decoder, Loss
from app.pytorch_pfn_extras import Trigger
from app.transforms.graph import Collate

level = logging.INFO
if sys.version_info[:2] >= (3, 8):
    # Python 3.8 or later
    logging.basicConfig(level=level, stream=sys.stdout,
                        force=True)
else:
    logging.root.handlers[0].setLevel(level)


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# dataset
parser.add_argument("--max-value", type=int, default=10)
# model
parser.add_argument("--channel", type=int, default=128)
# training
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--n-iter", type=int, default=2000)
parser.add_argument("--save-interval", type=int, default=500)
parser.add_argument("--out", type=str, required=True)
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
# debugging
parser.add_argument("--n-sample", type=int, default=0)
parser.add_argument("--use-eval", action="store_true")

args = parser.parse_args()

if args.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

# train dataset
logger.info("Initialize dataset")
dataset = RandomFlatDataset(np.random.RandomState(0), args.max_value)
collate = Collate(args.max_value)
if args.n_sample != 0:
    tmp = []
    for i, x in enumerate(dataset):
        tmp.append(x)
        if i + 1 == args.n_sample:
            break
    dataset = tmp

if args.use_eval:
    with open(os.path.join(
        os.path.dirname(__file__), "..", "datasets", "toy", "eval.yaml")
    ) as file:
        eval_dataset = yaml.load(file)
    parser = Parser()
    interpreter = Interpreter()
    dataset = []
    for sample in eval_dataset:
        name = sample["name"]
        code = sample["program"]
        program = parser.parse(code)
        examples = sample["examples"]
        for x in examples:
            example = Example(x["input"], x["output"])

            def eval_and_append(program: Program):
                if isinstance(program, Input):
                    out = example.inputs[program.id]
                    dataset.append(FlatSample(str(out), Example([], out)))
                    return out
                elif isinstance(program, Number) or isinstance(program, Boolean):
                    out = program.value
                    dataset.append(FlatSample(str(out), Example([], out)))
                    return out
                elif isinstance(program, Function):
                    inputs = []
                    for arg in program.args:
                        inputs.append(eval_and_append(arg))
                    out = interpreter.run(program, example.inputs)
                    dataset.append(FlatSample(program.name, Example(inputs, out)))
                    return out
                else:
                    raise AssertionError()
            eval_and_append(program)


# Module
logger.info("Initialize model")
model = Module(
    args.channel,
    collate.func,
    torch.nn.Linear(collate.value_encoder.vocab_size, args.channel),
    Decoder(args.channel, collate.value_encoder)
)
loss_fn = Loss()

optimizer = torch.optim.Adam([
    {
        "params": [
            v for k, v in model.named_parameters() if "bias" in k or "norm" in k
        ],
        "weight_decay": 0.0,
    },
    {
        "params": [
            v for k, v in model.named_parameters()
            if not ("bias" in k or "norm" in k)
        ],
        "weight_decay": 0.01,
    }
],
    lr=1e-4
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=args.n_iter,
)

# Training
logger.info("Initialize training utils")
loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=args.batch_size,
    collate_fn=collate,
    num_workers=4,
)
manager = ppe.training.ExtensionsManager(
    model, optimizer, args.n_iter,
    out_dir=args.out,
    extensions=[],
    iters_per_epoch=1,
)
manager.extend(
    extensions.FailOnNonNumber(),
    trigger=Trigger(args.save_interval, args.n_iter)
)
manager.extend(
    extensions.LogReport(
        trigger=Trigger(100, args.n_iter),
        filename="log.json",
    )
)
manager.extend(
    extensions.PrintReport(),
    trigger=Trigger(100, args.n_iter),
)
manager.extend(
    extensions.LRScheduler(scheduler),
    trigger=(1, "iteration"),
)

manager.extend(extensions.ProgressBar())
snapshot = extensions.snapshot(autoload=True, n_retains=1)
manager.extend(snapshot, trigger=Trigger(args.save_interval, args.n_iter))


def stop_trigger():
    return manager.stop_trigger or manager.iteration >= args.n_iter


logger.info("Start training")
while not stop_trigger():
    for p_func, p_args, _args, gt in loader:
        if stop_trigger():
            break
        with manager.run_iteration():
            model.train()
            p_func = p_func.to(device)
            p_args = p_args.to(device)
            _args = _args.to(device)
            gt = gt.to(device)
            out = model(p_func, _args, p_args)
            loss = loss_fn(out, gt)
            loss = loss.mean()

            loss.backward()
            optimizer.step()

            ppe.reporting.report({"loss": loss.item()})

torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
