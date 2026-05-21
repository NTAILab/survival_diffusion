from abc import abstractmethod
import torch
from ..util import Data
import numpy as np

class BaselineBase(torch.nn.Module):
    def __init__(self, output_dim: int, seed: int | None):
        super().__init__()
        self.output_dim = output_dim
        self.seed = int(np.random.default_rng(seed).integers(1, 2 ** 20))
        torch.manual_seed(self.seed)

    @abstractmethod
    def forward(self, data: Data) -> torch.Tensor:
        ...
