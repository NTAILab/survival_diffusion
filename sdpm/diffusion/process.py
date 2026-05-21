import torch

from ..feat_processing import DefaultFeatProcessor
from .scheduler import SchedulerBase
from .time_wrapper import TimeWrapperBase
import sys
from inspect import signature
from typing import Tuple, Dict
from ..util import Data, FeatNumber

class DDPM(torch.nn.Module):
    @staticmethod
    def _object_factory(device: str, obj_type: str, obj_name: str,
                       params: dict, seed: int | None):
        full_name = obj_name.title().replace("_", "") + obj_type.title().replace(
            "_", ""
        )
        base_cls_name = obj_type.title().replace("_", "") + "Base"
        cls = getattr(sys.modules[globals().get(base_cls_name).__module__], full_name)
        kw = params.copy()
        init_params = signature(cls.__init__).parameters
        if "device" in init_params:
            kw["device"] = device
        if "seed" in init_params:
            kw["seed"] = seed
        return cls(**kw)

    @staticmethod
    def obtain_type_params(config: dict):
        type = config["type"]
        params = config.copy()
        del params["type"]
        return type, params

    @staticmethod
    def get_scheduler(device: str, config) -> SchedulerBase:
        slr_type, slr_params = DDPM.obtain_type_params(config)
        return DDPM._object_factory(device, "scheduler", slr_type, slr_params, None)

    @staticmethod
    def get_time_wrapper(device: str, config, seed = None) -> TimeWrapperBase:
        tw_type, tw_params = DDPM.obtain_type_params(config)
        return DDPM._object_factory(device, "time_wrapper", tw_type, tw_params, seed)

    def __init__(
        self,
        device: torch.device,
        scheduler: SchedulerBase,
        time_wrapper: TimeWrapperBase,
        feat_processor: DefaultFeatProcessor,
        f_embed_noise: bool,
        loss: torch.nn.Module | None = None
    ):
        super().__init__()
        self.scheduler = scheduler.to(device)
        self.time_wrapper = time_wrapper.to(device)
        self.x_encoder = feat_processor.to(device)
        self.f_embed_noise = f_embed_noise
        if loss is None:
            self.loss = torch.nn.MSELoss()
        else:
            self.loss = loss.to(device)

    def get_mlp_input_dim(self) -> FeatNumber:
        encoder_dim = self.x_encoder.get_output_dim()
        time_dim = self.time_wrapper.get_output_dim()
        encoder_dim.time = time_dim
        if not self.f_embed_noise:
            encoder_dim.numeric += 2
        return encoder_dim

    @property
    def device(self):
        return self.time_wrapper.dev_tens.device

    def _get_mlp_input(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor | None,
        noise: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> Data:
        if self.f_embed_noise:
            X_num = torch.cat((noise, x_num), dim=-1)
        else:
            X_num = x_num
        X_proc = self.x_encoder(X_num, x_cat)
        if not self.f_embed_noise:
            X_proc.X_num = torch.cat((noise, X_proc.X_num), dim=-1)
        X_proc.time_emb = time_emb
        return X_proc

    # x is condition, T is target
    # returns diffusion MSE
    def forward_process(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor | None,
        E: torch.Tensor,  # event time
        D: torch.Tensor, # delta indicator
        mlp: torch.nn.Module,
    ):
        assert E.ndim == 2 and D.ndim == 2
        assert D.shape[-1] == 1, "D must be delta indicator "
        assert E.shape[-1] == 1, "Time tensor must contain only event times"
        Y = torch.cat((E, D), dim=-1)
        t_step = torch.randint(
            1, self.scheduler.time_steps_n,
            (E.shape[0], 1), device=self.device
        )
        noise = torch.randn_like(Y)
        c1 = torch.take_along_dim(self.scheduler.fw_mean, t_step.ravel(), dim=0).reshape_as(
            t_step
        )
        c2 = torch.take_along_dim(self.scheduler.fw_std, t_step.ravel(), dim=0).reshape_as(
            t_step
        )
        Y_noise = c1 * Y + c2 * noise
        time_embeddings = self.time_wrapper(t_step)
        mlp_in = self._get_mlp_input(x_num, x_cat, Y_noise, time_embeddings)
        noise_pred = mlp(mlp_in)
        if noise_pred.ndim == 3: # tabm
            noise = noise[:, None, :].repeat(1, noise_pred.shape[1], 1).reshape(-1, noise.shape[-1])
            noise_pred = noise_pred.reshape(-1, noise.shape[-1])
        assert noise_pred.shape == noise.shape
        return self.loss(noise_pred, noise)

    @torch.inference_mode()
    def reverse_process(
        self,
        x_num: torch.Tensor,
        x_cat: torch.Tensor | None,
        times_n: int,
        mlp: torch.nn.Module,
        seed: int | None = None,
    ):
        # assert x_num.ndim == 2 and (x_cat is None or x_cat.ndim == 2)
        if seed is not None:
            gen = torch.Generator(self.device).manual_seed(seed)
        else:
            gen = None
        y_t = torch.randn((times_n, 2), generator=gen, device=self.device)
        y_t = y_t.repeat(x_num.shape[0] // times_n, 1)
        ones = torch.ones(
            size=y_t.shape[:-1] + (1,), dtype=torch.long, device=self.device
        )
        for t in range(self.scheduler.time_steps_n - 1, 0, -1):
            t_step = t * ones
            time_emb = self.time_wrapper(t_step)
            mlp_in = self._get_mlp_input(x_num, x_cat, y_t, time_emb)
            pred_noise = mlp(mlp_in)
            if pred_noise.ndim == 3: # tabm
                pred_noise = torch.mean(pred_noise, dim=1)
            z = torch.randn(pred_noise.shape, device=self.device, generator=gen)
            c1, c2, c3 = (
                self.scheduler.rv_c1[t],
                self.scheduler.rv_c2[t],
                self.scheduler.rv_c3[t],
            )
            y_t = c1 * (y_t - c2 * pred_noise) + c3 * z  # check shapes
        return y_t
