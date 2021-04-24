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
        t0, t1, t2, t3, bsize = collate([x])
        assert t0.dtype == torch.int64
        assert t1.dtype == torch.bool
        assert t2.dtype == torch.float
        assert t3.dtype == torch.bool
        assert bsize == 1
        break
