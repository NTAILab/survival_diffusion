from abc import ABC, abstractmethod
import torch
import numpy as np

class TimeWrapperBase(torch.nn.Module, ABC):
    def __init__(self, device, seed):
        super().__init__()
        self.register_parameter('dev_tens', torch.nn.Parameter(torch.tensor([], device=device), requires_grad=False))
        self.seed = seed
        
    @abstractmethod
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_output_dim(self) -> int:
        pass
    
    @property
    def device(self):
        return next(self.parameters()).device

class TimeEmbeddingTimeWrapper(TimeWrapperBase):
    def __init__(self, device, emb_dim: int, time_steps_n: int, seed: int | None):
        super().__init__(device=device, seed=seed)
        self.t_emb = torch.nn.Embedding(time_steps_n, emb_dim).to(device)
        gen = torch.Generator(device).manual_seed(seed) if seed is not None else None
        torch.nn.init.normal_(self.t_emb.weight, mean=0, std=0.02, generator=gen)
        self.emb_dim = emb_dim

    def forward(self, t: torch.Tensor):
        emb = self.t_emb(t.ravel())
        emb = emb.reshape(t.shape[:-1] + (t.shape[-1] * emb.shape[-1],))
        return emb
    
    def get_output_dim(self) -> int:
        return self.emb_dim
    
class SinusoidalTimeWrapper(TimeWrapperBase):
    def __init__(self, device, emb_dim: int):
        assert emb_dim % 2 == 0 and emb_dim > 2, "Only even emb_dim > 2 are supported"
        super().__init__(device=device, seed=None)
        self.emb_dim = emb_dim
        self.register_buffer(
            "range",
            torch.arange(emb_dim // 2, dtype=torch.get_default_dtype(), device=device),
            persistent=False
        )

    def forward(self, t: torch.Tensor):
        half_dim = self.emb_dim // 2
        freqs = torch.exp(
            self.range * 
            -(np.log(10000) / (half_dim - 1))
        )
        args = t * freqs[None, :]
        emb = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)
        return emb
    
    def get_output_dim(self) -> int:
        return self.emb_dim
