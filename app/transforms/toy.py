from typing import List

import torch
from torchnlp.encoders import LabelEncoder

from app.datasets.toy import Parser, Sample


class Collate:
    def __init__(
        self,
        tokens: List[str],
        max_value: int,
        n_max_token_length: int,
        n_max_value_length: int,
    ):
        self.token_encoder = LabelEncoder(["<CLS>"] + tokens)
        values = [True, False, 0] + list(range(1, max_value + 1)) + \
            [-v for v in range(1, max_value + 1)]
        self.value_encoder = LabelEncoder([str(v) for v in values])
        self.max_value = max_value
        self.n_max_token_length = n_max_token_length
        self.n_max_value_length = n_max_value_length
        self.parser = Parser()

    def __call__(self, batch: List[Sample]):
        # transform
        batched_code = []
        batched_input = []
        batched_output = []
        for sample in batch:
            tokens = self.parser.unparse(sample.program).split(" ")
            tokens = ["<CLS>"] + tokens[:self.n_max_token_length]
            code = self.token_encoder.batch_encode(tokens)
            L = self.n_max_token_length + 1
            code = torch.nn.functional.pad(
                code, [0, L - len(code)]
            )
            code = torch.nn.functional.one_hot(
                code, num_classes=self.token_encoder.vocab_size
            ).float()
            for example in sample.examples:
                inputs = [str(v) for v in example.inputs[:self.n_max_value_length]]
                out = self.value_encoder.batch_encode([None] + inputs)  # None = unknown
                batched_code.append(code)
                batched_input.append(out)
                batched_output.append(str(example.output))

        # collate
        code = torch.nn.utils.rnn.pad_sequence(batched_code)
        input = torch.nn.utils.rnn.pad_sequence(batched_input)
        input_mask = torch.zeros(input.shape[0], input.shape[1], dtype=torch.bool)
        for i in range(len(batched_input)):
            input_mask[:len(batched_input[i]), i] = True
        output = self.value_encoder.batch_encode(batched_output)
        return code, input, input_mask, output, len(batch)
