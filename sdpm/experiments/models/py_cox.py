from typing import Tuple
from numpy import ndarray
import numpy as np
from numpy import recarray
from pycox.models import CoxPH, DeepHitSingle
from torchtuples.practical import MLPVanilla
from torchtuples.callbacks import EarlyStopping

import torch
from .base import CompetitivePycoxBase


def make_mlp(x_num_dim: int, x_cat_dim: int,
             n_hid_layers: int, hidden_dim: int, 
             norm: bool, output_dim: int,
             activation: str, dropout: float,
             output_bias: bool):
    nodes = [hidden_dim for _ in range(n_hid_layers)]
    return MLPVanilla(
        in_features=x_num_dim + x_cat_dim,
        num_nodes=nodes,
        out_features=output_dim,
        batch_norm=norm,
        dropout=dropout,
        activation=getattr(torch.nn, activation),
        output_bias=output_bias
    )
    

class DeepSurv(CompetitivePycoxBase):
    def __init__(self, device: str, x_num_dim: int, x_cat_dim: int,
                 n_hid_layers: int, hidden_dim: int,
                 norm: bool, activation: str, dropout: float,
                 lr: float, batch_size: int, epochs: int,
                 verbose: bool = False, seed: int | None = None):
        super().__init__(seed=seed)
        self.nn = make_mlp(
            x_num_dim=x_num_dim,
            x_cat_dim=x_cat_dim,
            n_hid_layers=n_hid_layers,
            hidden_dim=hidden_dim,
            norm=norm,
            output_dim=1,
            activation=activation,
            dropout=dropout,
            output_bias=False
        )
        self.model = CoxPH(self.nn, optimizer=torch.optim.Adam, device=device)
        self.model.optimizer.set_lr(lr)
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        
    def fit(self, X_num: ndarray, 
            X_cat: ndarray, y: np.recarray, 
            val_set: Tuple[ndarray, ndarray, recarray]):
        callbacks = [EarlyStopping()]
        x_train = np.concatenate((X_num, X_cat), axis=-1)
        self.y_train = y.copy()
        y = (y['time'].astype(np.float32).copy(), y["event"].astype(int).copy())
        val_set = (np.concatenate((val_set[0].astype(np.float32), val_set[1].astype(np.float32)), axis=-1), 
                   ((val_set[2]['time'].astype(np.float32).copy(), val_set[2]["event"].astype(int).copy())))
        self._fix_batch_size(x_train.shape[0])
        self.model.fit(x_train.astype(np.float32), y, self.batch_size, self.epochs, 
                       callbacks, self.verbose, val_data=val_set)
        self.model.compute_baseline_hazards()
    
    def predict_proba(self, X_num: ndarray, X_cat: ndarray):
        X_test = np.concatenate((X_num, X_cat), axis=-1)
        sf_df = self.model.predict_surv_df(X_test.astype(np.float32), batch_size=self.batch_size)
        sf = sf_df.to_numpy().T
        proba = np.empty_like(sf)
        proba[:, :-1] = sf[:, :-1] - sf[:, 1:]
        proba[:, -1] = sf[:, -1]
        return proba
        
class DeepHit(CompetitivePycoxBase):
    def __init__(self, device: str, x_num_dim: int,
                 x_cat_dim: int, time_ruler: np.ndarray,
                 n_hid_layers: int, hidden_dim: int,
                 norm: bool, activation: str, dropout: float,
                 lr: float, batch_size: int, epochs: int,
                 alpha: float, sigma: float,
                 verbose: bool = False, seed: int | None = None):
        super().__init__(seed=seed)
        self.labtrans = DeepHitSingle.label_transform(time_ruler)
        self.nn = make_mlp(
            x_num_dim=x_num_dim,
            x_cat_dim=x_cat_dim,
            n_hid_layers=n_hid_layers,
            hidden_dim=hidden_dim,
            norm=norm,
            output_dim=self.labtrans.out_features,
            activation=activation,
            dropout=dropout,
            output_bias=True
        )
        self.model = DeepHitSingle(self.nn, optimizer=torch.optim.Adam, 
                                   device=device, duration_index=self.labtrans.cuts,
                                   alpha=alpha, sigma=sigma)
        self.model.optimizer.set_lr(lr)
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        
    def fit(self, X_num: ndarray, 
            X_cat: ndarray, y: np.recarray, 
            val_set: Tuple[ndarray, ndarray, recarray]):
        callbacks = [EarlyStopping()]
        x_train = np.concatenate((X_num, X_cat), axis=-1)
        self.y_train = y.copy()
        min_uncens_time = y['time'][y["event"]].min()
        # thanks to labtrans.transform exception
        needed_idx = np.argwhere(y['time'] >= min_uncens_time).ravel()
        y = (y['time'][needed_idx].astype(np.float32), y["event"][needed_idx].astype(int).copy())
        x_train = x_train[needed_idx]
        y = self.labtrans.transform(*y)
        val_time = val_set[2]['time'].astype(np.float32).clip(min=self.labtrans.cuts.min() + 1e-6, max=self.labtrans.cuts.max() - 1e-6)
        val_event = val_set[2]["event"].astype(int).copy()
        y_val = self.labtrans.transform(val_time, val_event)
        val_set = (np.concatenate((val_set[0].astype(np.float32), val_set[1].astype(np.float32)), axis=-1), y_val)
        self._fix_batch_size(x_train.shape[0])
        self.model.fit(x_train.astype(np.float32), y, self.batch_size, self.epochs, 
                       callbacks, self.verbose, val_data=val_set)
