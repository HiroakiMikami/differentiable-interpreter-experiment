import torch
from torchnlp.encoders import LabelEncoder

from app.datasets.toy import FunctionName
from app.graph.infer import infer
from app.graph.module import Module
from app.nn.toy import Decoder, Loss


def test_infer():
    func_encoder = LabelEncoder(["True", "0", "1", "2", FunctionName.ID])
    value_encoder = LabelEncoder(["0", "1", "2"])
    module = Module(
        8,
        func_encoder,
        torch.nn.Linear(value_encoder.vocab_size, 8),
        Decoder(8, value_encoder)
    )
    assert infer(
        value_encoder.vocab_size,
        module,
        Loss(),
        1,
        [[1], [0], [True]],
        [1, 0, True],
        value_encoder,
        func_encoder,
        10,
        1,
        lambda x: True,
    ) is not None
