import numpy as np
import torch

from app.datasets.toy import RandomDataset
from app.transforms.toy import Collate


def test_collate():
    dataset = RandomDataset(
        np.random.RandomState(0),
        2,
        2,
        2,
        2,
    )
    collate = Collate(dataset.tokens, 2, 10, 2)
    for x in dataset:
        t0, t1, t2, out, bsize = collate([x])
        assert t0.dtype == torch.float32
        assert t0.shape == (11, 1, 27)
        assert t1.dtype == torch.float32
        assert t1.shape == (1, 1, 3)
        assert t2.dtype == torch.bool
        assert out.dtype == torch.float32
        assert out.shape == (1, 3)
        assert bsize == 1
        break
