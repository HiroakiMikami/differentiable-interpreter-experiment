import numpy as np
import torch
from torchnlp.encoders import LabelEncoder

from app.datasets.toy import Boolean, Function, FunctionName, Input, Number
from app.graph.graph import Interpreter, to_graph

func_encoder = LabelEncoder([
    "True",
    "0",
    FunctionName.ID,
    FunctionName.EQ,
])


def test_constant_to_graph():
    nodes = to_graph(func_encoder, Boolean(True), [], 1)
    assert len(nodes) == 1
    assert np.array_equal(nodes[0].p_func, [0, 1, 0, 0, 0])
    assert np.array_equal(nodes[0].p_args, torch.zeros(1, 0))

    nodes = to_graph(func_encoder, Number(0), [], 1)
    assert len(nodes) == 1
    assert np.array_equal(nodes[0].p_func, [0, 0, 1, 0, 0])
    assert np.array_equal(nodes[0].p_args, torch.zeros(1, 0))


def test_input_to_graph():
    nodes = to_graph(func_encoder, Input(0), [True], 1)
    assert len(nodes) == 1
    assert np.array_equal(nodes[0].p_func, [0, 1, 0, 0, 0])
    assert np.array_equal(nodes[0].p_args, torch.zeros(1, 0))

    nodes = to_graph(func_encoder, Input(0), [0], 1)
    assert len(nodes) == 1
    assert np.array_equal(nodes[0].p_func, [0, 0, 1, 0, 0])
    assert np.array_equal(nodes[0].p_args, torch.zeros(1, 0))


def test_function_to_graph():
    nodes = to_graph(
        func_encoder,
        Function(
            FunctionName.EQ,
            [
                Number(0),
                Function(FunctionName.ID, [Number(0)]),
            ]
        ),
        [],
    )
    # 0: Number(0)
    # 1: Number(0)
    # 2: ID(0)
    # 3: EQ(0, 0)
    assert len(nodes) == 4
    assert np.array_equal(nodes[0].p_func, [0, 0, 1, 0, 0])
    assert np.array_equal(nodes[0].p_args, torch.zeros(3, 0))
    assert np.array_equal(nodes[1].p_func, [0, 0, 1, 0, 0])
    assert np.array_equal(nodes[1].p_args, [[0], [0], [0]])
    assert np.array_equal(nodes[2].p_func, [0, 0, 0, 1, 0])
    assert np.array_equal(nodes[2].p_args, [[0, 1], [0, 0], [0, 0]])
    assert np.array_equal(nodes[3].p_func, [0, 0, 0, 0, 1])
    assert np.array_equal(nodes[3].p_args, [[1, 0, 0], [0, 0, 1], [0, 0, 0]])


def test_interpret_constant():
    def f(x, y, z):
        return torch.zeros(1, 3)

    nodes = to_graph(func_encoder, Boolean(True), [], 1)
    interpreter = Interpreter(f)
    out, raw = interpreter(nodes)
    assert np.allclose(out, [0.5, 0.5, 0])
    assert np.array_equal(raw, [0] * 3)
