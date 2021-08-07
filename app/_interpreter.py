import torch
from typing import List
from app._values import Values
from app._functions import Functions
from app.nn._generator import ValueGenerator, FunctionGenerator
from app.nn._function import Function, CompositeFunction


class Interpreter(torch.nn.Module):
    def __init__(
        self,
        z_dim: int,
        h_dim: int,
        c: int,
        max_value: int,
        function_arities: List[int],
    ) -> None:
        super().__init__()
        self.value_generator = ValueGenerator(z_dim, h_dim)
        self.function_generator = FunctionGenerator(z_dim, h_dim, len(function_arities))
        self.values = Values(self.value_generator, max_value)
        self.functions = Functions(self.function_generator)
        self.max_arity = max(function_arities)
        self.z_dim = z_dim

        function_impls = []
        for arity in function_arities:
            function_impls.append(Function(z_dim, c, arity))
        self.function_impl = CompositeFunction(function_impls)

    def forward(
        self,
        z_f: torch.Tensor,
        z_args: List[torch.Tensor],
    ) -> torch.Tensor:
        # f_z -> logit -> prob
        logit_f = self.function_generator(z_f)
        prob_f = torch.softmax(logit_f, dim=1)

        # apply function
        z_out = self.function_impl(prob_f, z_args)

        # normalize z_output
        self.values._normalize(z_out)
        return z_out

    def optimize_constants(self, n: int, lr: float = 1e-3) -> None:
        self.values.optimize(n, lr)
        self.functions.optimize(n, lr)
