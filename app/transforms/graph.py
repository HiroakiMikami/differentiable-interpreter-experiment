from typing import List

import torch
from torchnlp.encoders import LabelEncoder

from app.datasets.toy import FlatSample, FunctionName, Parser
from app.transforms.toy import encode_value


class Collate:
    def __init__(
        self,
        max_value: int,
    ):
        constants = [True, False, 0] + list(range(1, max_value + 1)) + \
            [-v for v in range(1, max_value + 1)]
        self.arities = {str(c): 0 for c in constants}
        for f in FunctionName.__members__.values():
            self.arities[f] = FunctionName.arity(f)
        self.func = LabelEncoder(list(self.arities.keys()))
        self.max_value = max_value
        self.parser = Parser()

    def __call__(self, batch: List[FlatSample]):
        n_func = self.func.vocab_size
        max_n_args = 0
        for sample in batch:
            max_n_args = max(max_n_args, len(sample.example.inputs))

        # transform
        batched_p_func = []
        batched_p_args = []
        batched_args = []
        batched_output = []
        for sample in batch:
            p_func = torch.full(size=(n_func,), fill_value=-1e10)
            p_func[self.func.encode(sample.function)] = 1e10
            batched_p_func.append(p_func)

            p_args = torch.full(size=(n_func, 3, max_n_args), fill_value=-1e10)
            args = torch.zeros(max_n_args, 3)
            for i, v in enumerate(sample.example.inputs):
                p_args[self.func.encode(sample.function), i, i] = 1e10
                args[i, :] = encode_value(v)
            batched_p_args.append(p_args)
            batched_args.append(args)
            batched_output.append(encode_value(sample.example.output))

        # collate
        p_func = torch.stack(batched_p_func)  # [N, n_func]
        p_args = torch.stack(batched_p_args)  # [N, 3, n_args]
        args = torch.stack(batched_args)  # [N, n_args, 3]
        output = torch.stack(batched_output, dim=0)  # [N, 3]
        return p_func, p_args, args, output
