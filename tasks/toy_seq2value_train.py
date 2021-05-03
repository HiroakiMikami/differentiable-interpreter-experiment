import argparse
import logging
import os
import sys

import numpy as np
import pytorch_pfn_extras as ppe
import torch
import yaml
from pytorch_pfn_extras.training import extensions

from app.datasets.toy import Example, Parser, RandomDataset, Sample
from app.nn.toy import Decoder, Loss
from app.pytorch_pfn_extras import Trigger
from app.seq2value.module import Module
from app.transforms.toy import Collate

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
parser.add_argument("--max-sample", type=int, default=2)
parser.add_argument("--max-input", type=int, default=8)
parser.add_argument("--max-value", type=int, default=10)
parser.add_argument("--max-depth", type=int, default=4)
parser.add_argument("--max-token-length", type=int, default=127)
# model
parser.add_argument("--channel", type=int, default=128)
parser.add_argument("--n-layer", type=int, default=6)
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
dataset = RandomDataset(np.random.RandomState(0), args.max_sample,
                        args.max_input, args.max_value, args.max_depth)
collate = Collate(dataset.tokens, args.max_value,
                  args.max_token_length, args.max_input)
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
    dataset = []
    for sample in eval_dataset:
        name = sample["name"]
        code = sample["program"]
        examples = sample["examples"]
        examples = [Example(x["input"], x["output"]) for x in examples]
        sample = Sample(parser.parse(code), examples)
        dataset.append(sample)


# Module
logger.info("Initialize model")
model = Module(
    args.channel, args.n_layer, collate.token_encoder.vocab_size,
    args.max_token_length, args.max_input,
    torch.nn.Embedding(collate.value_encoder.vocab_size, args.channel),
    Decoder(args.channel, collate.value_encoder),
)
loss_fn = Loss()
print(sum([x.numel() for x in model.parameters()]) / 1024 / 1024, "Mi")

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
    num_workers=2,
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
    for code, input, input_mask, gt, bsize in loader:
        if stop_trigger():
            break
        with manager.run_iteration():
            model.train()
            code = code.to(device)
            input = input.to(device)
            input_mask = input_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(code, input, input_mask)
            loss = loss_fn(out, gt)
            loss = loss.sum() / bsize

            loss.backward()
            optimizer.step()

            ppe.reporting.report({"loss": loss.item()})

torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
