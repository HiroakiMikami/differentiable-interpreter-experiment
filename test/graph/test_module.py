import numpy as np
import torch
from torchnlp.encoders import LabelEncoder

from app.datasets.toy import FlatDataset, FunctionName
from app.graph.module import GtModule, Module
from app.transforms.graph import Collate


def test_module_shape():
    module = Module(
        3, LabelEncoder([FunctionName.ID]), torch.nn.Linear(5, 3), torch.nn.Linear(3, 5)
    )
    out = module(
        torch.rand(2, 2),
        torch.rand(2, 7, 5),
        torch.rand(2, 2, 3, 7),
    )
    assert out.shape == (2, 5)


def test_arity_mask():
    module = Module(
        3, LabelEncoder([FunctionName.ID]), torch.nn.Linear(5, 3), torch.nn.Linear(3, 5)
    )
    p_func = torch.rand(2, 2)
    args = torch.rand(2, 2, 5)
    p_args = torch.rand(2, 2, 3, 2)
    with torch.no_grad():
        out0 = module(p_func, args, p_args)
        args[2:] = 0
        out1 = module(p_func, args, p_args)
    assert np.allclose(out0, out1)


def test_empty_arg():
    module = Module(
        3, LabelEncoder([FunctionName.ID]), torch.nn.Linear(5, 3), torch.nn.Linear(3, 5)
    )
    p_func = torch.rand(2, 2)
    args = torch.rand(2, 0, 5)
    p_args = torch.rand(2, 2, 3, 0)
    out = module(p_func, args, p_args)
    assert out.shape == (2, 5)


def test_gt_module():
    max_int = 10
    collate = Collate(max_int)
    module = GtModule(collate.func)
    dataset = FlatDataset(np.random.RandomState(0), max_int)
    for data in dataset:
        p_func, p_args, _args, gt = collate([data])
        gt = gt[0]
        raw = module(p_func, _args, p_args)[0]
        x = raw.clone()
        x[0] = torch.sigmoid(raw[0])
        x[1] = torch.sigmoid(raw[1])
        assert np.allclose(x[0], gt[0]), f"{data} {raw} {x} {gt}"
        if x[0] > 0.5:
            # bool
            assert np.allclose(x[1], gt[1]), f"{data} {raw} {x} {gt}"
        else:
            # int
            assert np.allclose(x[2], gt[2]), f"{data} {raw} {x} {gt}"
