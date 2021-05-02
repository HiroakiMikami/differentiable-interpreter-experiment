import numpy as np
import torch
from torchnlp.encoders import LabelEncoder

from app.datasets.toy import FunctionName
from app.graph.module import Module


def test_module_shape():
    module = Module(
        3, LabelEncoder([FunctionName.ID]), torch.nn.Linear(5, 3), torch.nn.Linear(3, 5)
    )
    out = module(
        torch.rand(2, 2),
        torch.rand(2, 7, 5),
        torch.rand(2, 3, 7),
    )
    assert out.shape == (2, 5)


def test_arity_mask():
    module = Module(
        3, LabelEncoder([FunctionName.ID]), torch.nn.Linear(5, 3), torch.nn.Linear(3, 5)
    )
    p_func = torch.rand(2, 2)
    args = torch.rand(2, 2, 5)
    p_args = torch.rand(2, 3, 2)
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
    p_args = torch.rand(2, 3, 0)
    out = module(p_func, args, p_args)
    assert out.shape == (2, 5)
