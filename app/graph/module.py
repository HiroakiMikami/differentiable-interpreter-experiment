import torch
from torchnlp.encoders import LabelEncoder

from app.datasets.toy import FunctionName
from app.transforms.toy import encode_value


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

        # normalize probs
        p_func = torch.softmax(p_func[:, 1:], dim=1)
        p_func = torch.cat([torch.zeros(N, 1, device=p_func.device), p_func], dim=1)
        p_args = torch.softmax(p_args, dim=2)

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


class GtModule(torch.nn.Module):
    def __init__(self, func_encoder: LabelEncoder):
        super().__init__()
        self.func_encoder = func_encoder

    def forward(self, p_func, args, p_args) -> torch.Tensor:
        """
        p_func: [N, n_func]
        args: [N, n_arg, E]
        p_args: [N, n_arity, n_arg]
        """
        N, _, E = args.shape

        # normalize probs
        p_func = torch.softmax(p_func[:, 1:], dim=1)
        p_func = torch.cat([torch.zeros(N, 1, device=p_func.device), p_func], dim=1)
        p_args = torch.softmax(p_args, dim=2)

        out = torch.zeros(N, E)
        for f in self.func_encoder.vocab:
            i = self.func_encoder.encode(f).item()
            p = p_func[:, i]
            p = p.reshape(-1, 1)
            if i == 0:
                continue
            if isinstance(f, str):
                out = out + p * self._constant(f)[None, :]
            else:
                out = out + p * self._func(f, args, p_args)
        return out

    def _constant(self, v: str) -> torch.Tensor:
        if v in set(["True", "False"]):
            out = encode_value(True if v == "True" else False)
        else:
            out = encode_value(int(v))
        out[0] = (out[0] * 1e10) * 2 - 1e10
        out[1] = (out[1] * 1e10) * 2 - 1e10
        return out

    def _func(
        self, func: FunctionName, args: torch.Tensor, p_args: torch.Tensor
    ) -> torch.Tensor:
        """
        args: [N, n_arg, E]
        p_args: [N, n_arity, n_arg]
        """
        N, _, E = args.shape
        out = torch.zeros(N, E)
        if args.numel() == 0:
            out[:, 0] = -1e10
            out[:, 2] = 1
            return out
        v0 = torch.sum(p_args[:, 0, :, None] * args, dim=1)
        v1 = torch.sum(p_args[:, 1, :, None] * args, dim=1)
        v2 = torch.sum(p_args[:, 2, :, None] * args, dim=1)
        if func == FunctionName.ID:
            out[:, 0] = (v0[:, 0] * 1e10) * 2 - 1e10
            out[:, 1] = (v0[:, 1] * 1e10) * 2 - 1e10
            out[:, 2] = v0[:, 2]
            return out
        elif func == FunctionName.NEG:
            out[:, 0] = (v0[:, 0] * 1e10) * 2 - 1e10
            out[:, 1] = 0
            out[:, 2] = -v0[:, 2]
            return out
        elif func == FunctionName.ADD:
            t = v0[:, 0] + v1[:, 0]
            out[:, 0] = (t * 1e10) * 2 - 1e10
            out[:, 1] = 0
            out[:, 2] = v0[:, 2] + v1[:, 2]
        elif func == FunctionName.SUB:
            t = v0[:, 0] + v1[:, 0]
            out[:, 0] = (t * 1e10) * 2 - 1e10
            out[:, 1] = 0
            out[:, 2] = v0[:, 2] - v1[:, 2]
        elif func == FunctionName.MUL:
            t = v0[:, 0] + v1[:, 0]
            out[:, 0] = (t * 1e10) * 2 - 1e10
            out[:, 1] = 0
            out[:, 2] = v0[:, 2] * v1[:, 2]
        elif func == FunctionName.DIV:
            t = v0[:, 0] + v1[:, 0]
            out[:, 0] = (t * 1e10) * 2 - 1e10
            out[:, 1] = 0
            d = torch.where(
                torch.abs(v1[:, 2]) < 1, torch.ones_like(v1[:, 2]), v1[:, 2]
            )
            out[:, 2] = v0[:, 2] / d
        elif func == FunctionName.MOD:
            t = v0[:, 0] + v1[:, 0]
            out[:, 0] = (t * 1e10) * 2 - 1e10
            out[:, 1] = 0
            d = torch.where(
                torch.abs(v1[:, 2]) < 1, torch.ones_like(v1[:, 2]), v1[:, 2]
            )
            out[:, 2] = torch.round(v0[:, 2]).long() % torch.round(d).long()
        elif func == FunctionName.NOT:
            out[:, 0] = v0[:, 0]
            out[:, 1] = -v0[:, 1]
            out[:, 2] = 0
        elif func == FunctionName.AND:
            t = v0[:, 0] + v1[:, 0]
            out[:, 0] = (t * 1e10) * 2 - 1e10
            x = torch.clamp(v0[:, 1] * v1[:, 1], -1e10, 1e10)
            out[:, 1] = (x * 1e10) * 2 - 1e10
            out[:, 2] = 0
        elif func == FunctionName.OR:
            t = v0[:, 0] + v1[:, 0]
            out[:, 0] = (t * 1e10) * 2 - 1e10
            x = torch.clamp(v0[:, 1] + v1[:, 1], -1e10, 1e10)
            out[:, 1] = (x * 1e10) * 2 - 1e10
            out[:, 2] = 0
        elif func == FunctionName.EQ:
            out[:, 0] = 1e10
            x = torch.round(v0[:, 2]) == torch.round(v1[:, 2])
            x = x.long()
            x = torch.clamp(x, -1e10, 1e10)
            out[:, 1] = (x * 1e10) * 2 - 1e10
            out[:, 2] = 0
        elif func == FunctionName.NE:
            out[:, 0] = 1e10
            x = torch.round(v0[:, 2]) != torch.round(v1[:, 2])
            x = x.long()
            x = torch.clamp(x, -1e10, 1e10)
            out[:, 1] = (x * 1e10) * 2 - 1e10
            out[:, 2] = 0
        elif func == FunctionName.LT:
            out[:, 0] = 1e10
            x = torch.round(v0[:, 2]) < torch.round(v1[:, 2])
            x = x.long()
            x = torch.clamp(x, -1e10, 1e10)
            out[:, 1] = (x * 1e10) * 2 - 1e10
            out[:, 2] = 0
        elif func == FunctionName.LE:
            out[:, 0] = 1e10
            x = torch.round(v0[:, 2]) <= torch.round(v1[:, 2])
            x = x.long()
            x = torch.clamp(x, -1e10, 1e10)
            out[:, 1] = (x * 1e10) * 2 - 1e10
            out[:, 2] = 0
        elif func == FunctionName.WHERE:
            p = v0[:, 1]
            t = v1[:, 0] + v2[:, 0]
            out[:, 0] = (t * 1e10) * 2 - 1e10
            out[:, 1] = 0
            out[:, 2] = p * v1[:, 2] + (1 - p) * v2[:, 2]
        return out
