import torch
from .diffusion.process import DDPM
from typing import Dict
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from itertools import chain
from typing import Dict, Tuple, List
from tqdm import tqdm
from .util import get_str_array, np2torch, score
from .baseline.base import BaselineBase
from copy import deepcopy
from .naming import Metric

class SDPM:
    def __init__(self, baseline: BaselineBase,
                 diffusion: DDPM,
                 epochs_n: int,
                 val_period: int,
                 batch_size: int,
                 optimizer: str,
                 optimizer_params: Dict,
                 f_log_time: bool,
                 f_gauss_delta: bool,
                 delta_scale: float = 1,
                 verbose: int = 1,
                 patience: int | None = None,
                 warmup: int | None = None,
                 val_metric: Metric = "c_index",
                 seed: int | None = None):
        self.diffusion = diffusion
        self.baseline: torch.nn.Module = baseline.to(self.device)
        self.delta_scale = delta_scale
        self.epochs_n = epochs_n
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.optimizer_params = optimizer_params
        self.val_period = val_period
        self.processed_epochs = 0
        self.verbose = verbose
        self.y_train = None
        self.cur_ptc = 0
        self.patience = patience
        self.warmup = warmup
        self.val_metric = val_metric
        self.best_metric = None
        self.bl_params = None
        self.df_params = None
        self.time_grid = None
        self.tau_mean = None
        self.tau_sigma = None
        self.f_log_time = f_log_time
        self.f_gauss_delta = f_gauss_delta
        self.loss_hist = None
        self.val_hist = None
        self.seed = seed if seed is not None else np.random.randint(1, 2 ** 20)
        
    def _resolve_patience(self, train_n: int):
        if self.warmup is None:
            self.warmup = 100
        if train_n < 500:
            if self.patience is None:
                self.patience = 20
        elif train_n < 2500:
            if self.patience is None:
                self.patience = 10
        else:
            if self.patience is None:
                self.patience = 5
        
    @property
    def device(self):
        return self.diffusion.device
    
    def _get_optimizer(self, model: torch.nn.Module):
        optim_cls = getattr(torch.optim, self.optimizer)
        params = chain(model.parameters(), self.diffusion.parameters())
        return optim_cls(params=params, **self.optimizer_params)

    def _clip_gradients(self) -> int:
        try:
            torch.nn.utils.clip_grad_norm_(
                chain(self.baseline.parameters(), self.diffusion.parameters()),
                1.0,
                foreach=self.device.type != 'mps',
                error_if_nonfinite=True
            )
            return 0
        except RuntimeError as e:
            print(f'NaN in gradients! Error: {e}')
            count = 0
            for param in chain(self.baseline.parameters(), self.diffusion.parameters()):
                if param.grad is None:
                    continue
                mask = ~torch.isfinite(param.grad)
                count += int(torch.count_nonzero(mask).item())
                param.grad[mask] = 0
            return count
    
    @torch.inference_mode()
    def _helper_predict(self, x_num: np.ndarray,
                x_cat: np.ndarray | None,
                times_n: int,
                batch_size: int,
                seed: int | None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = times_n * max(1, batch_size // times_n)
        X_num = np2torch(x_num, device=self.device)
        X_cat = np2torch(x_cat, device=self.device, dtype=torch.long) if x_cat is not None else None
        get_ds = lambda x_num, x_cat=None: TensorDataset(x_num) if x_cat is None else TensorDataset(x_num, x_cat)
        ds_all = get_ds(X_num, X_cat)
        dl_all = DataLoader(ds_all, batch_size=512, shuffle=False)
        pred_raw_time_list = []
        pred_raw_c_list = []
        for data_outer in dl_all:
            repeater = lambda t: t[:, None, :].repeat(1, times_n, 1).flatten(end_dim=1)
            data_outer = map(repeater, data_outer)
            ds_inner = get_ds(*data_outer)
            dl_inner = DataLoader(ds_inner, batch_size=batch_size, shuffle=False)
            for inner_batch in dl_inner:
                x_num_cur = inner_batch[0]
                x_cat_cur = inner_batch[1] if len(inner_batch) > 1 else None
                pred_raw = self.diffusion.reverse_process(
                    x_num=x_num_cur,
                    x_cat=x_cat_cur,
                    times_n=times_n,
                    mlp=self.baseline,
                    seed=seed if seed is not None else self.seed,
                )
                pred_raw_time_list.append(pred_raw[..., 0])
                pred_raw_c_list.append(pred_raw[..., 1])
        t_raw_all = torch.cat(pred_raw_time_list, dim=0).reshape(-1, times_n)
        c_raw_all = torch.cat(pred_raw_c_list, dim=0).reshape(-1, times_n)
        t_pred = self._inverse_transform_t(t_raw_all)
        c_pred = self._inverse_transform_latent_d(c_raw_all)
        return t_pred, c_pred

    @torch.inference_mode()
    def predict(self, x_num: np.ndarray, 
                x_cat: np.ndarray | None,
                times_n: int,
                batch_size: int = 32768,
                seed: int | None = None) -> np.recarray:
        T, C = self._helper_predict(x_num, x_cat, times_n, batch_size, seed)
        T = T.cpu().numpy()
        C = C.cpu().numpy()
        return get_str_array(T, C)
    
    @torch.inference_mode()
    def predict_sf(self, x_num: np.ndarray,
                   x_cat: np.ndarray | None,
                   times_n: int,
                   time_grid: np.ndarray | None,
                   batch_size: int = 32768,
                   seed: int | None = None) -> np.ndarray:
        T, C = self._helper_predict(x_num, x_cat, times_n, batch_size, seed)
        if time_grid is not None:
            time_grid = np2torch(np.unique(time_grid), device=self.device)
        S = self._get_sf(T, C, time_grid=time_grid)
        return S.cpu().numpy()

    def _get_sf(self, T: torch.Tensor,
                C: torch.Tensor,
                eps: float = 1e-6,
                time_grid: torch.Tensor | None = None) -> torch.Tensor:
        assert T.ndim == 2 and C.ndim == 2 and T.shape == C.shape
        if time_grid is None:
            time_grid = self.time_grid

        B, N = T.shape
        device = self.device

        t_sorted, order = torch.sort(T, dim=1)
        c_sorted = torch.gather(C, 1, order).to(dtype=T.dtype)

        is_new = torch.ones((B, N), dtype=torch.bool, device=device)
        is_new[:, 1:] = torch.abs(t_sorted[:, 1:] - t_sorted[:, :-1]) > eps
        group_id = torch.cumsum(is_new.to(torch.long), dim=1) - 1

        ones = torch.ones((B, N), dtype=T.dtype, device=device)

        group_size = torch.zeros((B, N), dtype=T.dtype, device=device)
        group_size.scatter_add_(1, group_id, ones)

        d = torch.zeros((B, N), dtype=T.dtype, device=device)
        d.scatter_add_(1, group_id, c_sorted)

        t_group = torch.full((B, N), float("inf"), dtype=T.dtype, device=device)
        t_group.scatter_(1, group_id, t_sorted)

        prefix = torch.cumsum(group_size, dim=1) - group_size
        Y = (N - prefix).to(dtype=T.dtype)
        Y = torch.clamp(Y, min=eps)

        frac = d / Y
        frac = torch.clamp(frac, 0.0, 1.0 - eps)
        log_step = torch.log1p(-frac)
        logS = torch.cumsum(log_step, dim=1)
        S = torch.exp(logS)

        idx = torch.searchsorted(t_group, time_grid[None, :].repeat(B, 1), side='right') - 1
        idx_clamped = torch.clamp(idx, min=0)

        S_grid = torch.gather(S, 1, idx_clamped)
        S_grid[idx < 0] = 1
        return S_grid
    
    def get_exp_time(self, sf: np.ndarray, sorted_time: np.ndarray):
        t = sorted_time
        integral = t[:, 0] + np.sum(sf[:, :-1] * (t[:, 1:] - t[:, :-1]), axis=-1)
        return integral
    
    def score(self, X_num: np.ndarray,
              X_cat: np.ndarray | None,
              y: np.recarray,
              times_n: int = 1024, batch_size: int = 32768,
              seed: int | None = None) -> Tuple[float, float, float]:
        assert not self.baseline.training
        T, C = self._helper_predict(X_num, X_cat, times_n, batch_size, seed)
        sf = self._get_sf(T, C)
        return score(self.y_train,
                     sf.cpu().numpy(),
                     self.time_grid.cpu().numpy(),
                     y)

    def _validate(self, epoch: int,
                  val_set: Tuple[np.ndarray, np.ndarray | None, np.recarray]):
        if val_set is None or (epoch - 1) % self.val_period != 0 or epoch <= self.warmup:
            return None, None, None
        self.baseline.eval()
        self.diffusion.eval()
        c_index, ibs, auc = self.score(*val_set, times_n=256)
        self.baseline.train()
        self.diffusion.train()
        return c_index, ibs, auc
    
    def _save_params(self):
        self.bl_params = deepcopy(self.baseline.state_dict())
        self.df_params = deepcopy(self.diffusion.state_dict())
        
    def _restore_params(self):
        if self.bl_params is None:
            return
        self.baseline.load_state_dict(self.bl_params)
        self.diffusion.load_state_dict(self.df_params)
    
    def _patience_mechanism(self,
                            c_index: float | None,
                            ibs: float | None,
                            auc: float | None):
        metrics = {
            "c_index": c_index,
            "ibs": ibs,
            "auc": auc,
        }
        
        current_metric = metrics[self.val_metric]
        if current_metric is None or not np.isfinite(current_metric):
            return
        if self.best_metric is None:
            improved = True
        elif self.val_metric == "ibs":
            improved = current_metric < self.best_metric
        else:
            improved = current_metric > self.best_metric
        if improved:
            self.best_metric = current_metric
            self._save_params()
            self.cur_ptc = 0
        else:
            self.cur_ptc += 1

    def _transform_t(self, T: torch.Tensor) -> torch.Tensor:
        if self.f_log_time:
            log = torch.log(T.clamp_min(1e-7))
        else:
            log = T
        sigma, mean = torch.std_mean(log, dim=0, keepdim=True, unbiased=False)
        self.tau_mean = mean
        sigma[sigma < 1e-8] = 1
        self.tau_sigma = sigma
        return (log - self.tau_mean) / self.tau_sigma

    def _inverse_transform_t(self, tau: torch.Tensor) -> torch.Tensor:
        if self.f_log_time:
            tau_restored = tau * self.tau_sigma + self.tau_mean
            t = torch.exp(tau_restored)
            return t
        else:
            tau_restored = tau * self.tau_sigma + self.tau_mean
            return tau_restored.clamp_min(0.0)

    def _generate_latent_d(self, delta: torch.Tensor) -> torch.Tensor:
        mu_0 = -1 * self.delta_scale
        mu_1 = 1 * self.delta_scale
        sigma = 0.25 * self.delta_scale
        mu = torch.where(
            delta == 0,
            torch.full_like(delta, mu_0, dtype=torch.get_default_dtype()),
            torch.full_like(delta, mu_1, dtype=torch.get_default_dtype())
        )
        if self.f_gauss_delta:
            eps = torch.randn_like(mu)
            return mu + sigma * eps
        else:
            return torch.where(
                        delta == 0,
                        torch.full_like(delta, -1.0, dtype=torch.get_default_dtype()),
                        torch.full_like(delta, 1.0, dtype=torch.get_default_dtype())
                    )

    def _inverse_transform_latent_d(self, z: torch.Tensor) -> torch.Tensor:
        d = torch.where(
            condition=z > 0,
            input=torch.ones(size=z.shape, dtype=torch.long, device=z.device),
            other=torch.zeros(size=z.shape, dtype=torch.long, device=z.device)
        )
        return d

    def fit(self, X_num: np.ndarray, X_cat: np.ndarray | None,
            y: np.recarray, val_set: Tuple[np.ndarray, np.ndarray | None, np.recarray] | None):
        # assert X_num.ndim == 2 and (X_cat is None or X_cat.ndim == 2) and y.ndim == 1
        self._resolve_patience(X_num.shape[0])
        torch.manual_seed(self.baseline.seed)
        self.baseline.train()
        self.diffusion.train()
        self.y_train = y.copy()
        self.time_grid = np2torch(np.unique(y['time']), device=self.device)
        X_n_t = np2torch(X_num, self.device)
        X_c_t = None if X_cat is None else np2torch(X_cat, self.device, torch.long)
        t_t = self._transform_t(np2torch(y['time'].copy()[:, None], self.device))
        delta_t = np2torch(y["event"].copy()[:, None], device=self.device, dtype=torch.long)
        optimizer = self._get_optimizer(self.baseline)
        self.cur_ptc, self.best_metric = 0, None
        self._save_params()
        self.loss_hist = []
        self.val_hist = {
            'c_index': [],
            'ibs': [],
            'auc': [],
        }
        val_postfix = lambda metric: ' (ES metric)' if metric == self.val_metric else ''
        for ep in range(1, self.epochs_n + 1):
            ds = TensorDataset(X_n_t, t_t, delta_t) if X_cat is None else TensorDataset(X_n_t, X_c_t, t_t, delta_t)
            dl = DataLoader(ds, self.batch_size, True)
            if self.verbose > 0 and (ep % self.verbose == 0 or ep == 1):
                desc = f'Epoch {ep}'
                if ep <= self.warmup:
                    desc += ' (warmup)'
                bar = tqdm(iterable=dl, desc=desc)
                cur_verb = True
            else:
                bar = dl
                cur_verb = False
            cum_loss = 0
            for i, data in enumerate(bar, 1):
                if len(data) == 3:
                    x_n_b, t_b, d_b = data
                    x_c_b = None
                else:
                    x_n_b, x_c_b, t_b, d_b = data
                z_b = self._generate_latent_d(d_b)
                # if ep == 1 and i == 1:
                #     print(z_b)
                loss = self.diffusion.forward_process(
                    x_num=x_n_b,
                    x_cat=x_c_b,
                    E=t_b,
                    D=z_b,
                    mlp=self.baseline
                )
                if not torch.isfinite(loss):
                    print('Non-finite loss!')
                    continue
                cum_loss += loss.item()
                if cur_verb:
                    bar.set_postfix_str(f'MSE: {cum_loss / i:.3}')
                loss.backward()
                self._clip_gradients()
                optimizer.step()
                optimizer.zero_grad()
            self.loss_hist.append(cum_loss / i)
            val_c_ind, val_ibs, val_auc = self._validate(ep, val_set)
            self._patience_mechanism(val_c_ind, val_ibs, val_auc)
            
            if val_c_ind is not None:
                self.val_hist['c_index'].append(val_c_ind)
                self.val_hist['ibs'].append(val_ibs)
                self.val_hist['auc'].append(val_auc)
                
            if val_c_ind is not None and self.verbose > 0:
                print(f'Validation C-index{val_postfix('c_index')}:', val_c_ind)
                print(f'Validation IBS{val_postfix('ibs')}:', val_ibs)
                print(f'Validation AUC{val_postfix('auc')}:', val_auc)
                print('Patience:', self.cur_ptc)
            if val_set is not None and self.cur_ptc == self.patience:
                if self.verbose:
                    print('Early stopping!')
                break
        self._restore_params()
        self.baseline.eval()
        self.diffusion.eval()
