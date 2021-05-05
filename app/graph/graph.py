from dataclasses import dataclass
from typing import List, Union

import torch
from torchnlp.encoders import LabelEncoder

from app.datasets.toy import Boolean, Function, Input, Number, Program

# , FunctionName


@dataclass
class Node:
    p_func: torch.Tensor
    p_args: torch.Tensor


def to_graph(
    func_encoder: LabelEncoder,
    p: Program,
    inputs: List[Union[bool, int]],
    n_max_arity: int = 3,
    n_args: int = 0,
) -> List[Node]:
    p_func = torch.zeros(func_encoder.vocab_size)
    if isinstance(p, Input):
        v = inputs[p.id]
        if isinstance(v, bool):
            p = Boolean(v)
        else:
            p = Number(v)
        return to_graph(func_encoder, p, [], n_max_arity, n_args)
    elif isinstance(p, Number) or isinstance(p, Boolean):
        p_args = torch.zeros(n_max_arity, n_args)
        p_func[func_encoder.encode(str(p.value))] = 1
        return [Node(p_func, p_args)]
    elif isinstance(p, Function):
        nodes = []
        arg_idx = []
        # eval arguments
        for arg in p.args:
            arg_nodes = to_graph(func_encoder, arg, inputs, n_max_arity, n_args)
            n_args += len(arg_nodes)
            nodes.extend(arg_nodes)
            arg_idx.append(n_args - 1)
        p_args = torch.zeros(n_max_arity, n_args)
        p_func[func_encoder.encode(p.name)] = 1
        for i, idx in enumerate(arg_idx):
            p_args[i, idx] = 1
        nodes.append(Node(p_func, p_args))
        return nodes
    else:
        raise AssertionError(f"Invalid program: {p}")


class Interpreter(torch.nn.Module):
    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, nodes: List[Node]) -> torch.Tensor:
        assert len(nodes) > 0
        args = torch.zeros(0, 3)  # [n_arg, 3]
        raw = None
        for node in nodes:
            out = self.module(
                node.p_func.unsqueeze(0),
                args.unsqueeze(0),
                node.p_args.unsqueeze(0),
            )[0]  # [3]
            raw = out
            out = out.clone()
            out[0] = torch.sigmoid(out[0])
            out[1] = torch.sigmoid(out[1])
            args = torch.cat([args, out.unsqueeze(0)], dim=0)  # [n_arg + 1, 3]
        return args[-1], raw
