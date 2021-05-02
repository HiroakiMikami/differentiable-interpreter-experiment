import torch
from torchnlp.encoders import LabelEncoder

from app.seq2value.infer import infer
from app.seq2value.module import Module


def test_infer():
    encoder = LabelEncoder(["0", "1", "2"])
    module = Module(
        8, 1, encoder.vocab_size, 80, 8, torch.nn.Embedding(5, 8), torch.nn.Identity()
    )
    encoded_input = torch.randint(0, 5, size=(5, 1))
    input_mask = torch.randint(0, 1, size=(5, 1)).bool()
    output = torch.rand(3, 8)
    assert infer(
        module,
        torch.nn.MSELoss(),
        10,
        encoder,
        encoded_input,
        input_mask,
        output,
        10,
        1,
        lambda x: True,
    ) is not None
