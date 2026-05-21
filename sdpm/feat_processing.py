from abc import ABC, abstractmethod
import torch
from .util import Data, FeatNumber

class NumericProcessorBase(torch.nn.Module, ABC):
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features

    @abstractmethod
    def get_output_dim(self) -> FeatNumber:
        ...

    def _check_input(self, x_num: torch.Tensor):
        if x_num.ndim != 2:
            raise RuntimeError("Input must have 2 dimensions")
        if x_num.shape[-1] != self.n_features:
            raise RuntimeError(f"Input must contain {self.n_features} numeric features")
        
class CategoricalProcessorBase(torch.nn.Module, ABC):
    def __init__(self, n_features: int):
        super().__init__()
        self.n_features = n_features
        
    @abstractmethod
    def get_output_dim(self) -> FeatNumber:
        ...
        
    def _check_input(self, x_cat: torch.Tensor | None):
        if x_cat is None:
            return
        if x_cat.ndim != 2:
            raise RuntimeError("Input must have 2 dimensions")
        if x_cat.dtype != torch.long:
            # check for 0s and 1s is computationally long...
            raise RuntimeError("Categorical features must be one-hot int64 tensor")
        if x_cat.shape[-1] != self.n_features:
            raise RuntimeError(f"Input must contain {self.n_features} categorical features")

class FourierFeatProcessor(NumericProcessorBase):
    def __init__(self, n_features: int, n_frequencies: int,
            frequency_init_scale: float):
        super().__init__(n_features)
        self.init_scale = frequency_init_scale
        self.weight = torch.nn.Parameter(torch.empty(n_features, n_frequencies))
        bound = frequency_init_scale * 3
        torch.nn.init.trunc_normal_(self.weight, 0.0,
            frequency_init_scale, a=-bound, b=bound)
        self.n_features = n_features

    def _periodic_forward(self, x: torch.Tensor) -> torch.Tensor:
        x = 2 * torch.pi * self.weight[None, ...] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1)
        return x

    def forward(self, x_num: torch.Tensor) -> Data:
        self._check_input(x_num)
        X = self._periodic_forward(x_num) # (batch_n, task_n, num_feat_n, 2 * freq_n)
        return Data(X_num=X.flatten(-2, -1)) # (batch_n, task_n, total_emb_dim)

    def get_output_dim(self) -> FeatNumber:
        return FeatNumber(numeric=self.weight.shape[0] * self.weight.shape[1] * 2)
    
class EmptyNumericProcessor(NumericProcessorBase):
    def __init__(self, n_features: int):
        super().__init__(n_features)
        
    def forward(self, x_num: torch.Tensor):
        self._check_input(x_num)
        return Data(X_num=x_num, X_cat=None)

    def get_output_dim(self) -> FeatNumber:
        return FeatNumber(self.n_features, 0)

class DefaultFeatProcessor(torch.nn.Module):
    def __init__(self, num_proc: NumericProcessorBase,
                 cat_proc: CategoricalProcessorBase):
        super().__init__()
        self.num_proc = num_proc
        self.cat_proc = cat_proc

    def forward(self, x_num: torch.Tensor, 
                x_cat: torch.Tensor | None) -> dict:
        X_num_p = self.num_proc(x_num)
        X_cat_p = self.cat_proc(x_cat)
        if X_cat_p.X_num is not None:
            X_num_t = torch.cat((X_num_p.X_num, X_cat_p.X_num), dim=-1)
        else:
            X_num_t = X_num_p.X_num
        return Data(X_num=X_num_t, X_cat=X_cat_p.X_cat)

    def get_output_dim(self) -> FeatNumber:
        return self.num_proc.get_output_dim() + self.cat_proc.get_output_dim()

class EmptyCatProcessor(CategoricalProcessorBase):
    def __init__(self, n_features: int):
        super().__init__(n_features)

    def forward(self, x_cat: torch.Tensor | None):
        self._check_input(x_cat)
        return Data(X_num=None, X_cat=x_cat)

    def get_output_dim(self) -> FeatNumber:
        return FeatNumber(0, self.n_features)
    
class EmbeddingCatProcessor(CategoricalProcessorBase):
    def __init__(self, n_features: int, emb_dim: int):
        super().__init__(n_features)
        self.embeddings = torch.nn.Embedding(2 * n_features, emb_dim)
        self.register_buffer(
            name='shift',
            tensor=2 * torch.arange(0, n_features, dtype=torch.long),
            persistent=False
        )

    def forward(self, x_cat: torch.Tensor | None):
        if x_cat is None:
            return Data(X_num=None, X_cat=None)
        self._check_input(x_cat)
        E = self.embeddings(self.shift[None, :] + x_cat).reshape((x_cat.shape[0], -1))
        return Data(X_num=E, X_cat=None)

    def get_output_dim(self) -> FeatNumber:
        return FeatNumber(self.n_features * self.embeddings.embedding_dim, 0)
