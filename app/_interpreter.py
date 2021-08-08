import torch
from app.nn._function import Function
from app.nn._feature_extractor import FeatureExtractor


class Interpreter(torch.nn.Module):
    def __init__(
        self,
        z_dim: int,
        c: int,
        max_arity: int,
    ) -> None:
        super().__init__()
        self.value_extractor = FeatureExtractor(1, z_dim, c)
        self.function_extractor = FeatureExtractor(1, z_dim, c)
        self.function_impl = Function(z_dim, c, max_arity)
        self.max_arity = max_arity
        self.z_dim = z_dim

    def forward(self, *args, **kwargs):
        raise NotImplementedError
