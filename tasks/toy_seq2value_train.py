import argparse
import logging
import os
import sys

import numpy as np
import pytorch_pfn_extras as ppe
import torch
from pytorch_pfn_extras.training import extensions

from app.datasets.toy import RandomDataset
from app.nn.model import Model
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
    logging.root.handlers[0].setLevel(level)


def main():
    logger = logging.Logger(__name__)

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
    parser.add_argument("--n-iter", type=int, default=10000)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    # debugging
    parser.add_argument("--n-sample", type=int, default=0)

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

    # Module
    logger.info("Initialize model")
    encoder = Module(
        args.channel, args.n_layer, collate.token_encoder.vocab_size,
        args.max_token_length, args.max_input, torch.nn.Linear(3, args.channel)
    )
    decoder = Decoder(args.channel, args.max_value)
    model = Model(encoder, decoder)
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

    manager.extend(extensions.ProgressBar())
    snapshot = extensions.snapshot(autoload=True, n_retains=1)
    manager.extend(snapshot, trigger=Trigger(args.save_interval, args.n_iter))

    logger.info("Start training")
    while not manager.stop_trigger:
        for code, code_mask, input, input_mask, gt, bsize in loader:
            if manager.stop_trigger:
                break
            with manager.run_iteration():
                model.train()
                code = code.to(device)
                code_mask = code_mask.to(device)
                input = input.to(device)
                input_mask = input_mask.to(device)

                optimizer.zero_grad(set_to_none=True)
                out = model(code, code_mask, input, input_mask)
                loss = loss_fn(out, gt)
                loss = loss.sum() / bsize

                loss.backward()
                optimizer.step()

                ppe.reporting.report({"loss": loss.item()})

    torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))


if __name__ == "__main__":
    main()