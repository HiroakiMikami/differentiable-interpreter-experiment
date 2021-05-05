from typing import Callable, List, Optional, Union

import torch
from torchnlp.encoders import LabelEncoder
from tqdm import trange

from app.datasets.toy import Boolean, Function, FunctionName, Input, Number, Program
from app.graph.graph import Interpreter, Node
from app.transforms.toy import encode_value


def uniform_nodes(n_node: int, func_encoder: LabelEncoder) -> List[Node]:
    nodes = []
    for i in range(n_node):
        p_func = torch.full((func_encoder.vocab_size,), 1.0 / func_encoder.vocab_size)
        p_func.requires_grad = True
        # TODO max arity
        p_args = torch.full((3, i), 1.0 / i if i != 0 else 0.0)
        p_args.requires_grad = True
        nodes.append(Node(p_func, p_args))
    return nodes


def infer(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    max_node: int,
    inputs: List[List[Union[int, bool]]],
    outputs: List[Union[int, bool]],
    func_encoder: LabelEncoder,
    n_optimize: int,
    check_interval: int,
    validate: Callable[[Program], bool],
    lr: float = 0.1,
) -> Optional[Program]:
    with torch.no_grad():
        interpreter = Interpreter(model)
        n_node = max_node + len(inputs[0])

        nodes = uniform_nodes(n_node, func_encoder)

        # encode output
        gt = []
        for o in outputs:
            gt.append(encode_value(o))

        # convert inputs to constant
        graphs = []
        for xs in inputs:
            tmp = nodes
            for i, x in enumerate(xs):
                tmp[i] = Node(tmp[i].p_func.clone(), tmp[i].p_args.clone())
                tmp[i].p_func[:] = 0.0
                tmp[i].p_func[func_encoder.encode(str(x))] = 1.0
                tmp[i].p_args[:] = 0.0  # arity == 0
            graphs.append(tmp)

    params = []
    for node in nodes[len(inputs[0]):]:
        params.append(node.p_func)
        params.append(node.p_args)

    for i in trange(n_optimize):
        # calc gradient of p_func and p_args
        optimizer = torch.optim.SGD(params, lr=lr)
        optimizer.zero_grad()
        loss = torch.zeros(())
        for j, graph in enumerate(graphs):
            _, out = interpreter(graph)
            loss = loss + loss_fn(out.reshape(1, -1), gt[j].reshape(1, -1)).sum()
        loss = loss / len(graph)
        loss.backward()
        optimizer.step()

        # normalize prob
        with torch.no_grad():
            for node in nodes[len(inputs[0]):]:
                node.p_func[:] = torch.softmax(node.p_func, dim=0)
                node.p_args[:] = torch.softmax(node.p_args, dim=1)

        if (i + 1) % check_interval == 0:
            # decode nodes
            results = [Input(i) for i in range(len(inputs[0]))]
            for node in nodes[len(inputs[0]):]:
                f = torch.argmax(node.p_func[1:], dim=0)  # exclude <unk>
                func = func_encoder.decode(f + 1)
                if isinstance(func, FunctionName):
                    if node.p_args.numel() == 0:
                        # invalid
                        print(f"invalid: {func} for the first node")
                        break
                    args = []
                    args_tensor = torch.argmax(node.p_args, dim=1)  # [max_arity]
                    for i in range(FunctionName.arity(func)):
                        arg = args_tensor[i].item()
                        args.append(results[arg])
                    results.append(Function(func, args))
                else:
                    # constant
                    if func in set(["True", "False"]):
                        # bool
                        results.append(Boolean(True if func == "True" else False))
                    else:
                        # int
                        results.append(Number(int(func)))

            if len(results) != 0:
                p = results[-1]
                print(p)
                if validate(p):
                    return p
    return None
