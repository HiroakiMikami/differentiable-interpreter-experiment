import argparse
import logging
import os
import sys

import numpy as np
import torch
import yaml

from app.datasets.toy import Example, Interpreter, Parser, RandomDataset, Sample
from app.nn.toy import Decoder, Loss
from app.seq2value.infer import infer
from app.seq2value.module import Module
from app.transforms.toy import Collate

level = logging.INFO
logger = logging.getLogger(__name__)
logger.setLevel(level)
if sys.version_info[:2] >= (3, 8):
    # Python 3.8 or later
    logging.basicConfig(level=level, stream=sys.stdout,
                        force=True)
else:
    logging.basicConfig(level=level, stream=sys.stdout)
    # logging.root.handlers[0].setLevel(level)


parser = argparse.ArgumentParser()
# dataset
parser.add_argument("--max-sample", type=int, default=2)
parser.add_argument("--max-input", type=int, default=8)
parser.add_argument("--max-value", type=int, default=10)
parser.add_argument("--max-depth", type=int, default=4)
parser.add_argument("--max-token-length", type=int, default=127)
# model
parser.add_argument("--channel", type=int, default=128)
parser.add_argument("--n-layer", type=int, default=6)
parser.add_argument("--model-path", type=str, required=True)
# evaluation
parser.add_argument("--task", type=str, default="pbe", choices=["interpreter", "pbe"])
parser.add_argument("--n-optimize", type=int, default=1000)
parser.add_argument("--check-interval", type=int, default=100)
parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])


args = parser.parse_args()

if args.device == "cpu":
    device = torch.device("cpu")
else:
    device = torch.device("cuda")

# train dataset
logger.info("Initialize dataset")
dataset = RandomDataset(np.random.RandomState(0), args.max_sample,
                        args.max_input, args.max_value, args.max_depth)
collate = Collate(dataset.tokens, args.max_value, args.max_token_length, args.max_input)


with open(os.path.join(
    os.path.dirname(__file__), "..", "datasets", "toy", "eval.yaml")
) as file:
    eval_dataset = yaml.load(file)

# Module
logger.info("Initialize model")
model = Module(
    args.channel, args.n_layer, collate.token_encoder.vocab_size,
    args.max_token_length, args.max_input,
    torch.nn.Linear(3, args.channel),
    Decoder(args.channel),
)
model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
loss_fn = Loss()

parser = Parser()
interpreter = Interpreter()

# evaluation
for sample in eval_dataset:
    name = sample["name"]
    code = sample["program"]
    examples = sample["examples"]
    examples = [Example(x["input"], x["output"]) for x in examples]
    sample = Sample(parser.parse(code), examples)
    (
        encoded_code, encoded_input, input_mask, encoded_output, _
    ) = collate([sample])

    logger.info(f"eval for {name}")
    if args.task == "pbe":
        def validate(tokens):
            code = " ".join(tokens)
            program = parser.parse(code)
            if program is None:
                return False
            for example in examples:
                pred = interpreter.run(program, example.inputs)
                if pred != example.output:
                    return False
            return True

        out = infer(
            model, loss_fn, args.max_token_length, collate.token_encoder, encoded_input,
            input_mask, encoded_output, args.n_optimize, args.check_interval,
            validate,
        )
        if out is None:
            logger.info("  Fail to synthesize")
        else:
            code = " ".join(out)
            logger.info(f"  {code}")
    elif args.task == "interpreter":
        with torch.no_grad():
            preds = model(encoded_code, encoded_input, input_mask)
            loss = loss_fn(preds, encoded_output)
        for i, example in enumerate(examples):
            pred = preds[i]
            is_bool = torch.sigmoid(pred[0])
            is_true = torch.sigmoid(pred[1])
            if is_bool >= 0.5:
                # bool
                pred_output = True if is_true >= 0.5 else False
            else:
                pred_output = round(pred[2].item())

            logger.info(
                f" {i}-th Example: {example.inputs} => " +
                f"gt={example.output} pred={pred_output} " +
                f"(loss={loss[i]})"
            )
