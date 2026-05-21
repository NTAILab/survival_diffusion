from abc import ABC, abstractmethod
import torch
import numpy as np

class SchedulerBase(torch.nn.Module, ABC):
    def __init__(self, device: torch.device,
                 time_steps_n: int,
                 alphas: torch.Tensor,
                 alphas_cumpord: torch.Tensor,
                 betas: torch.Tensor):
        super().__init__()
        self._time_steps_n = time_steps_n
        
        def cdt(t: torch.Tensor):
            return t.to(device=device, dtype=torch.get_default_dtype())
        
        self._fw_mean = cdt(torch.sqrt(alphas_cumpord))
        self._fw_std = torch.sqrt(1 - alphas_cumpord)
        denom = self._fw_std.clone()
        denom[denom < 1e-15] = 1
        self._rv_c2 = cdt((1 - alphas) / denom)
        self._fw_std = cdt(self._fw_std)
        self._rv_c1 = cdt(1 / torch.sqrt(alphas))
        self._rv_c3 = torch.sqrt((1 - alphas_cumpord[:-1]) / (1 - alphas_cumpord[1:]) * betas[1:])
        self._rv_c3 = cdt(torch.cat((torch.zeros((1,)), self._rv_c3)))
        # sqrt(alpha_bar)
        self.register_buffer('fw_mean', self._fw_mean, persistent=False)
        # sqrt(1 - alpha_bar)
        self.register_buffer('fw_std', self._fw_std, persistent=False)
        # 1 / sqrt(alpha)
        self.register_buffer('rv_c1', self._rv_c1, persistent=False)
        # (1 - alpha) / sqrt(1 - alpha_bar)
        self.register_buffer('rv_c2', self._rv_c2, persistent=False)
        # sqrt(beta)
        self.register_buffer('rv_c3', self._rv_c3, persistent=False)

    @property
    def time_steps_n(self):
        return self._time_steps_n

class CosineScheduler(SchedulerBase):
    def __init__(self, device: torch.device, time_steps_n: int, s: float = 0.008):
        t = torch.arange(0, time_steps_n, dtype=torch.float64)
        alphas_cumprod = torch.cos(((t / time_steps_n) + s) / (1 + s) * np.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        alphas = torch.empty_like(alphas_cumprod)
        alphas[0] = 1
        alphas[1:] = alphas_cumprod[1:] / alphas_cumprod[:-1]
        betas = 1 - alphas
        super().__init__(device=device, time_steps_n=time_steps_n, alphas=alphas, alphas_cumpord=alphas_cumprod,
                         betas=betas)
