from typing import List

import torch
from torchnlp.encoders import LabelEncoder

from app.datasets.toy import FlatSample, FunctionName, Parser


class Collate:
    def __init__(
        self,
        max_value: int,
    ):
        constants = [True, False, 0] + list(range(1, max_value + 1)) + \
            [-v for v in range(1, max_value + 1)]
        self.arities = {c: 0 for c in constants}
        for f in FunctionName.__members__:
            self.arities[f] = FunctionName.arity(f)
        self.func = LabelEncoder(list(self.arities.keys()))
        self.value_encoder = LabelEncoder(
            [True, False, 0] + list(range(1, max_value + 1)) +
            [-v for v in range(1, max_value + 1)]
        )
        self.max_value = max_value
        self.parser = Parser()

    def __call__(self, batch: List[FlatSample]):
        max_n_args = 0
        for sample in batch:
            max_n_args = max(max_n_args, len(sample.example.inputs))

        # transform
        batched_p_func = []
        batched_p_args = []
        batched_args = []
        batched_output = []
        for sample in batch:
            batched_p_func.append(
                torch.nn.functional.one_hot(
                    self.func.encode(sample.function),
                    num_classes=self.func.vocab_size,
                ).float()
            )

            p_args = torch.zeros(3, max_n_args)
            args = torch.zeros(max_n_args, self.value_encoder.vocab_size)
            for i, v in enumerate(sample.example.inputs):
                p_args[i, i] = 1.0
                args[i, :] = torch.nn.functional.one_hot(
                    self.value_encoder.encode(v),
                    num_classes=self.value_encoder.vocab_size
                )
            batched_p_args.append(p_args)
            batched_args.append(args)
            batched_output.append(sample.example.output)

        # collate
        p_func = torch.stack(batched_p_func)  # [N, n_func]
        p_args = torch.stack(batched_p_args)  # [N, 3, n_args]
        args = torch.stack(batched_args)  # [N, n_args, n_value]
        output = self.value_encoder.batch_encode(batched_output)
        return p_func, p_args, args, output
