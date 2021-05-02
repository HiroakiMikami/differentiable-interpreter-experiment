import torch
from torchnlp.encoders import LabelEncoder

from app.datasets.toy import FunctionName


class Module(torch.nn.Module):
    def __init__(
        self,
        C: int,
        func_encoder: LabelEncoder,
        encoder: torch.nn.Module,
        decoder: torch.nn.Module,
    ):
        super().__init__()
        arities = []
        for x in func_encoder.vocab:
            if isinstance(x, FunctionName):
                arities.append(FunctionName.arity(x))
            else:
                arities.append(0)

        self.n_func = len(arities)
        self.arities = arities
        self.encoder = encoder
        self.decoder = decoder

        for i, arity in enumerate(arities):
            if arity == 0:
                p = torch.nn.Parameter(torch.zeros(C))
                torch.nn.init.normal_(p)
                self.register_parameter(f"func{i}", p)
            else:
                module = torch.nn.Sequential(
                    torch.nn.Linear(arity * C, 4 * arity * C),
                    torch.nn.Tanh(),
                    torch.nn.Linear(4 * arity * C, 4 * C),
                    torch.nn.Tanh(),
                    torch.nn.Linear(4 * C, C)
                )
                self.add_module(f"func{i}", module)

    def forward(self, p_func, args, p_args):
        """
        p_func: [N, n_func]
        args: [N, n_arg, E]
        p_args: [N, n_arity, n_arg]
        """
        N, n_func = p_func.shape
        N, n_arity, n_arg = p_args.shape
        args = self.encoder(args)  # [N, n_arg, C]
        N, n_arg, C = args.shape
        args = args.reshape(N, 1, n_arg, C)
        p_args = p_args.reshape(N, n_arity, n_arg, 1)
        args = p_args * args  # [N, n_arity, n_arg, C]
        args = torch.sum(args, dim=2)  # [N, n_arity, C]

        modules = dict(self.named_modules())
        params = dict(self.named_parameters())
        _out = []
        for i, arity in enumerate(self.arities):
            k = f"func{i}"
            if arity == 0:
                _out.append(params[k].reshape(1, -1).expand(N, -1))
            else:
                a = args[:, :arity, :]  # [N, arity, C]
                a = a.reshape(N, -1)
                _out.append(modules[k](a))
        # _out: list of [N, C]
        out = torch.stack(_out, dim=1)  # [N, n_func, C]
        p_func = p_func.reshape(N, n_func, 1)
        out = p_func * out  # [N, n_func, C]
        out = torch.sum(out, dim=1)  # [N, C]
        return self.decoder(out)
