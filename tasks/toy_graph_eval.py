import argparse
import logging
import os
import sys

import torch
import yaml

from app.datasets.toy import Example, Interpreter, Parser, Sample
from app.graph.graph import Interpreter as InterpreterModule
from app.graph.graph import to_graph
from app.graph.infer import infer
from app.graph.module import GtModule, Module
from app.nn.toy import Decoder, Loss
from app.transforms.graph import Collate

level = logging.INFO
if sys.version_info[:2] >= (3, 8):
    # Python 3.8 or later
    logging.basicConfig(level=level, stream=sys.stdout,
                        force=True)
else:
    logging.basicConfig(level=level, stream=sys.stdout)
    # logging.root.handlers[0].setLevel(level)


logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
# dataset
parser.add_argument("--max-value", type=int, default=10)
# model
parser.add_argument("--channel", type=int, default=128)
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--max-nodes", type=int, default=8)
# evaluation
parser.add_argument("--task", type=str, default="pbe", choices=["interpreter", "pbe"])
parser.add_argument("--n-optimize", type=int, default=1000)
parser.add_argument("--check-interval", type=int, default=10)
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
# other
parser.add_argument("--gt", action="store_true")

args = parser.parse_args()

if args.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

with open(os.path.join(
    os.path.dirname(__file__), "..", "datasets", "toy", "eval.yaml")
) as file:
    eval_dataset = yaml.load(file)

collate = Collate(args.max_value)

# Module
logger.info("Initialize model")
if args.gt:
    model = GtModule(collate.func)
else:
    model = Module(
        args.channel,
        collate.func,
        torch.nn.Linear(3, args.channel),
        Decoder(args.channel)
    )
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
interpreter_module = InterpreterModule(model)
loss_fn = Loss()

parser = Parser()
interpreter = Interpreter()

# evaluation
for sample in [eval_dataset[1]]:
    name = sample["name"]
    code = sample["program"]
    examples = sample["examples"]
    examples = [Example(x["input"], x["output"]) for x in examples]
    sample = Sample(parser.parse(code), examples)

    logger.info(f"eval for {name}")
    if args.task == "pbe":
        def validate(program):
            if program is None:
                return False
            for example in examples:
                pred = interpreter.run(program, example.inputs)
                if pred != example.output:
                    return False
            return True

        out = infer(
            model,
            loss_fn,
            args.max_nodes,
            [example.inputs for example in examples],
            [example.output for example in examples],
            collate.func,
            args.n_optimize,
            args.check_interval,
            validate,
        )
        if out is None:
            logger.info("  Fail to synthesize")
        else:
            code = parser.unparse(out)
            logger.info(f"  {code}")
    elif args.task == "interpreter":
        for i, example in enumerate(sample.examples):
            nodes = to_graph(collate.func, sample.program, example.inputs)
            with torch.no_grad():
                pred, raw = interpreter_module(nodes)
                # loss = loss_fn(logit, encoded_outputs[i].unsqueeze(0))
                is_bool = pred[0]
                if is_bool >= 0.5:
                    # bool
                    is_true = pred[1]
                    pred_output = True if is_true >= 0.5 else False
                else:
                    pred_output = round(pred[2].item())

            logger.info(
                f" {i}-th Example: {example.inputs} => " +
                f"gt={example.output} pred={pred_output}"
            )
