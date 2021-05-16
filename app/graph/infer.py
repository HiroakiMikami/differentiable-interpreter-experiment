import tempfile
from typing import Callable, List, Optional, Union

import pytorch_pfn_extras as ppe
import torch
from pytorch_pfn_extras.training import extensions
from torchnlp.encoders import LabelEncoder

from app.datasets.toy import Boolean, Function, FunctionName, Input, Number, Program
from app.graph.graph import Interpreter, Node
from app.pytorch_pfn_extras import Trigger
from app.transforms.toy import encode_value


def uniform_nodes(n_node: int, func_encoder: LabelEncoder) -> List[Node]:
    nodes = []
    for i in range(n_node):
        p_func = torch.zeros(func_encoder.vocab_size)
        p_func.requires_grad = True
        # TODO max arity
        p_args = torch.zeros(func_encoder.vocab_size, 3, i)
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
) -> Optional[Program]:
    lrs = [1]  # [1e-1, 1e0, 1e1, 1e2, 1e4, 1e8]
    for lr in lrs:
        print(f"LR={lr}")
        x = _infer(
            model, loss_fn, max_node, inputs, outputs, func_encoder,
            n_optimize, check_interval, validate, lr
        )
        if x is not None:
            return x
    return None


def _infer(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    max_node: int,
    inputs: List[List[Union[int, bool]]],
    outputs: List[Union[int, bool]],
    func_encoder: LabelEncoder,
    n_optimize: int,
    check_interval: int,
    validate: Callable[[Program], bool],
    lr: float,
) -> Optional[Program]:
    kldiv = torch.nn.KLDivLoss()

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
                tmp[i].p_func[:] = -1e10
                tmp[i].p_func[func_encoder.encode(str(x))] = 1e10
                tmp[i].p_args[:] = 0.0
            graphs.append(tmp)

    params = []
    for node in nodes[len(inputs[0]):]:
        params.append(node.p_func)
        params.append(node.p_args)

    # optimizer = torch.optim.Adam(params, lr=lr)
    optimizer = torch.optim.Adam(
        params,
        lr=lr,
        betas=(0.5, 0.999),
        weight_decay=1e-3,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        manager = ppe.training.ExtensionsManager(
            model, optimizer, n_optimize / check_interval,
            out_dir=tmpdir,
            extensions=[],
            iters_per_epoch=check_interval,
        )
        manager.extend(
            extensions.FailOnNonNumber(),
            trigger=Trigger(check_interval, n_optimize)
        )
        manager.extend(
            extensions.LogReport(
                trigger=Trigger(check_interval, n_optimize),
                filename="log.json",
            )
        )
        manager.extend(
            extensions.PrintReport(),
            trigger=Trigger(check_interval, n_optimize),
        )
        manager.extend(extensions.ProgressBar())

        for i in range(n_optimize):
            # calc gradient of p_func and p_args
            with manager.run_iteration():
                optimizer.zero_grad()
                outloss = torch.zeros(())
                klloss = torch.zeros(())
                pred_outputs = []
                with torch.autograd.detect_anomaly():
                    for j, graph in enumerate(graphs):
                        for node in graph:
                            klloss = klloss + kldiv(
                                torch.log(torch.clamp_min(node.p_func, 1e-5)),
                                torch.full_like(node.p_func, 1.0 / node.p_func.shape[0])
                            )
                            if node.p_args.shape[1] != 0:
                                klloss = klloss + kldiv(
                                    torch.log(
                                        torch.clamp_min(
                                            node.p_args.permute(0, 2, 1), 1e-5
                                        )
                                    ),
                                    torch.full_like(
                                        node.p_args, 1.0 / node.p_args.shape[1]
                                    ).permute(0, 2, 1)
                                )
                        pred, out = interpreter(graph)
                        outloss = outloss + loss_fn(
                            out.reshape(1, -1), gt[j].reshape(1, -1)
                        ).sum()
                        is_bool = pred[0]
                        if is_bool >= 0.5:
                            # bool
                            is_true = pred[1]
                            pred_output = True if is_true >= 0.5 else False
                        else:
                            if torch.isinf(pred[2]) or torch.isnan(pred[2]):
                                pred_output = None
                            else:
                                pred_output = round(pred[2].item())
                        pred_outputs.append(pred_output)
                    outloss = outloss / len(graph)
                    loss = outloss  # - klloss
                    loss.backward()
                torch.nn.utils.clip_grad_value_(params, 0.1)
                grad_norm = 0
                for p in params:
                    if p.numel() == 0:
                        continue
                    grad_norm += torch.norm(p.grad.reshape(-1), dim=0).mean()
                ppe.reporting.report({"grad_norm": grad_norm / len(params)})
                with torch.no_grad():
                    for i, v in enumerate(func_encoder.vocab):
                        print(v)
                        print(f"  {nodes[-1].p_func.grad[i].item()}")
                        for j, p in enumerate(nodes[-1].p_args.grad[i]):
                            print(f"  {j}-th arg: {p} ({nodes[-1].p_args[i, j]})")
                    print("---")

                optimizer.step()
                ppe.reporting.report({"loss": loss.item()})
                with torch.no_grad():
                    # Add smoothing
                    smoothing = 0.01
                    for node in nodes[len(inputs[0]):]:
                        p = torch.softmax(node.p_func[1:], dim=0)
                        p[:] += smoothing / p.shape[0]
                        p[p.argmax(dim=0)] -= smoothing
                        node.p_func[1:] = torch.log(p / (1 - p))
                        if node.p_args.numel() == 0:
                            continue
                        node.p_func[:] = torch.clamp(node.p_func, -1e10, 1e10)

                        # [n_func, max_arity, n_arg]
                        p = torch.softmax(node.p_args, dim=2)
                        p[:] += smoothing / p.shape[2]
                        m = p.argmax(dim=2)
                        for k in range(func_encoder.vocab_size):
                            for j in range(3):
                                p[k, j, m[k, j]] -= smoothing
                        node.p_args[:] = torch.log(p / (1 - p))
                        node.p_args[:, :, :] = torch.clamp(node.p_args, -1e10, 1e10)

                if (i + 1) % check_interval == 0:
                    print("Synthesize: ")
                    print(f"  preds={pred_outputs}")
                    print(f"  GT={outputs}")
                    # decode nodes
                    results = [Input(i) for i in range(len(inputs[0]))]
                    for node in nodes[len(inputs[0]):]:
                        # print(node.p_func[1:])
                        # print(node.p_func[1:][f])
                        f = torch.argmax(node.p_func[1:], dim=0)  # exclude <unk>

                        # Discretize p_func
                        """
                        with torch.no_grad():
                            node.p_func[:] = 0.0
                            node.p_func[f + 1] = 1.0
                        """

                        func = func_encoder.decode(f + 1)
                        if isinstance(func, FunctionName):
                            if node.p_args.numel() == 0:
                                # invalid
                                print(f"invalid: {func} for the first node")
                                break
                            args = []
                            # [n_func, max_arity]
                            args_tensor = torch.argmax(node.p_args, dim=2)
                            for i in range(FunctionName.arity(func)):
                                arg = args_tensor[f + 1, i].item()
                                args.append(results[arg])

                                # Discretize p_args
                                """
                                with torch.no_grad():
                                    node.p_args[i, :] = 0
                                    node.p_args[i, args_tensor[i].item()] = 1.0
                                """
                            results.append(Function(func, args))
                        else:
                            # constant
                            if func in set(["True", "False"]):
                                # bool
                                results.append(
                                    Boolean(True if func == "True" else False)
                                )
                            else:
                                # int
                                results.append(Number(int(func)))

                    if len(results) != 0:
                        p = results[-1]
                        print(p)
                        if validate(p):
                            return p
        return None
