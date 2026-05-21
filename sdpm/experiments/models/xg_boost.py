from typing import Tuple

import numpy as np
import xgboost as xgb
from xgbse import XGBSEKaplanNeighbors, XGBSEStackedWeibull
from lifelines.exceptions import ConvergenceError
from numpy import ndarray, recarray
from scipy.stats import norm

from .base import CompetitiveModelBase

from ...util import score


class AFTModel(CompetitiveModelBase):
    def __init__(
        self,
        num_boost_round: int,
        time_ruler: np.ndarray,
        early_stopping_rounds: int = 50,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        l2_reg: float = 0.01,
        l1_reg: float = 0.01,
        tree_method: str = "hist",
        max_bin: int = 256,
        aft_loss_distribution: str = "normal",
        aft_loss_distribution_scale: float = 1.0,
        n_jobs: int | None = None,
        seed: int | None = None,
        verbose: bool = False,
    ):
        super().__init__()
        self.params = {
            "verbosity": int(verbose),
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "tree_method": tree_method,
            "learning_rate": learning_rate,
            "aft_loss_distribution": aft_loss_distribution,
            "aft_loss_distribution_scale": aft_loss_distribution_scale,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "lambda": l2_reg,  # L2 regularization.
            "alpha": l1_reg,  # L1 regularization.
            "max_bin": max_bin,
            "seed": self.seed,
        }
        if n_jobs is not None:
            self.params["nthread"] = n_jobs
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.time_ruler = time_ruler.copy()
        self.y_train = None

    def _get_xgb_set(self, X_num: np.ndarray, X_cat: np.ndarray | None, y: recarray | None):
        get_full_x = (
            lambda x_num, x_cat: x_num if x_cat is None else np.concatenate((x_num, x_cat), axis=-1)
        )
        data = xgb.DMatrix(get_full_x(X_num, X_cat))
        if y is not None:
            y_lb = y["time"].copy().clip(min=1e-6)
            y_ub = y_lb.copy()
            y_ub[y["event"] == 0] = np.inf
            data.set_float_info("label_lower_bound", y_lb)
            data.set_float_info("label_upper_bound", y_ub)
        return data

    def _aft_survival_from_prediction(self, pred: np.ndarray):
        sigma = self.params["aft_loss_distribution_scale"]
        dist = self.params["aft_loss_distribution"]
        z = (np.log(self.time_ruler)[None, :] - pred[:, None]) / sigma

        if dist == "normal":
            return 1.0 - norm.cdf(z)
        elif dist == "logistic":
            cdf = 1.0 / (1.0 + np.exp(-z))
            return 1.0 - cdf
        elif dist == "extreme":
            return np.exp(-np.exp(z))
        else:
            raise ValueError(f"Unknown AFT dist: {dist}")

    def fit(
        self,
        X_num: ndarray,
        X_cat: ndarray | None,
        y: recarray,
        val_set: Tuple[ndarray, ndarray | None, recarray] | None = None,
    ):
        self.y_train = y.copy()
        train_data = self._get_xgb_set(X_num, X_cat, y)
        evals = []
        callbacks = []
        if val_set is not None:
            val_data = self._get_xgb_set(*val_set)
            evals.append((val_data, "valid"))
            callbacks.append(
                xgb.callback.EarlyStopping(
                    rounds=self.early_stopping_rounds,
                    save_best=True,
                    maximize=False,
                    data_name="valid",
                    metric_name="aft-nloglik",
                )
            )

        train_kwargs = {
            "params": self.params,
            "dtrain": train_data,
            "num_boost_round": self.num_boost_round,
            "evals": evals,
            "verbose_eval": bool(self.params["verbosity"]),
            "callbacks": callbacks,
        }

        self.bst = xgb.train(**train_kwargs)

    def score(self, X_num: ndarray, X_cat: ndarray | None, y: recarray) -> Tuple[float, float, float]:
        test_data = self._get_xgb_set(X_num, X_cat, None)
        pred = np.asarray(self.bst.predict(test_data, output_margin=True))
        sf = self._aft_survival_from_prediction(pred)
        return score(self.y_train, sf, self.time_ruler, y)

class GBMWeibull(CompetitiveModelBase):
    def __init__(
        self,
        num_boost_round: int,
        time_ruler: np.ndarray,
        early_stopping_rounds: int = 50,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        l2_reg: float = 0.01,
        l1_reg: float = 0.01,
        tree_method: str = "hist",
        max_bin: int = 256,
        aft_loss_distribution: str = "normal",
        aft_loss_distribution_scale: float = 1.0,
        wb_penalty: float = 0.0,
        wb_l1_ratio: float = 0.1,
        n_jobs: int | None = None,
        seed: int | None = None,
        verbose: bool = False):
        super().__init__(seed=seed)
        self.gbm_params = {
            "verbosity": int(verbose),
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "tree_method": tree_method,
            "learning_rate": learning_rate,
            "aft_loss_distribution": aft_loss_distribution,
            "aft_loss_distribution_scale": aft_loss_distribution_scale,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "lambda": l2_reg,  # L2 regularization.
            "alpha": l1_reg,  # L1 regularization.
            "max_bin": max_bin,
            "seed": self.seed,
        }
        if n_jobs is not None:
            self.gbm_params["nthread"] = n_jobs
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.time_ruler = time_ruler.copy()
        self.y_train = None
        self.verbose = verbose
        self.model = XGBSEStackedWeibull(
            xgb_params=self.gbm_params,
            weibull_params={
                "penalizer": wb_penalty,
                "l1_ratio": wb_l1_ratio
            }
        )
        
    def fit(
        self,
        X_num: ndarray,
        X_cat: ndarray | None,
        y: recarray,
        val_set: Tuple[ndarray, ndarray | None, recarray] | None = None,
    ):
        self.y_train = y.copy()
        X = np.concatenate((X_num, X_cat), axis=-1) if X_cat is not None else X_num
        if val_set is not None:
            val_set_fixed = (
                np.concatenate((val_set[0], val_set[1]), axis=-1) if X_cat is not None else val_set[0],
                val_set[2]
            )
        else:
            val_set_fixed = None
        try:
            self.model.fit(
                X=X,
                y=y,
                num_boost_round=self.num_boost_round,
                validation_data=val_set_fixed,
                early_stopping_rounds=self.early_stopping_rounds,
                verbose_eval=self.verbose,
                persist_train=False,
                time_bins=self.time_ruler
            )
        except TypeError as e:
            # convergence error actually, Infs appeared during boosting bulding
            raise ConvergenceError(str(e))

    def score(self, X_num: ndarray, X_cat: ndarray | None, y: recarray) -> Tuple[float, float, float]:
        X = np.concatenate((X_num, X_cat), axis=-1) if X_cat is not None else X_num
        sf = self.model.predict(X).to_numpy()
        return score(self.y_train, sf, self.time_ruler, y)
    
class GBMKM(CompetitiveModelBase):
    def __init__(
        self,
        num_boost_round: int,
        time_ruler: np.ndarray,
        early_stopping_rounds: int = 50,
        learning_rate: float = 0.05,
        max_depth: int = 5,
        min_child_weight: float = 1.0,
        subsample: float = 1.0,
        colsample_bytree: float = 1.0,
        gamma: float = 0.0,
        l2_reg: float = 0.01,
        l1_reg: float = 0.01,
        tree_method: str = "hist",
        max_bin: int = 256,
        aft_loss_distribution: str = "normal",
        aft_loss_distribution_scale: float = 1.0,
        n_neighbors: int = 30,
        n_jobs: int | None = None,
        seed: int | None = None,
        verbose: bool = False):
        super().__init__(seed=seed)
        self.gbm_params = {
            "verbosity": int(verbose),
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "tree_method": tree_method,
            "learning_rate": learning_rate,
            "aft_loss_distribution": aft_loss_distribution,
            "aft_loss_distribution_scale": aft_loss_distribution_scale,
            "max_depth": max_depth,
            "min_child_weight": min_child_weight,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "gamma": gamma,
            "lambda": l2_reg,  # L2 regularization.
            "alpha": l1_reg,  # L1 regularization.
            "max_bin": max_bin,
            "seed": self.seed,
        }
        if n_jobs is not None:
            self.gbm_params["nthread"] = n_jobs
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.time_ruler = time_ruler.copy()
        self.y_train = None
        self.verbose = verbose
        self.model = XGBSEKaplanNeighbors(
            xgb_params=self.gbm_params,
            n_neighbors=n_neighbors
        )
        
    def fit(
        self,
        X_num: ndarray,
        X_cat: ndarray | None,
        y: recarray,
        val_set: Tuple[ndarray, ndarray | None, recarray] | None = None,
    ):
        self.y_train = y.copy()
        X = np.concatenate((X_num, X_cat), axis=-1) if X_cat is not None else X_num
        if val_set is not None:
            val_set_fixed = (
                np.concatenate((val_set[0], val_set[1]), axis=-1) if X_cat is not None else val_set[0],
                val_set[2]
            )
        else:
            val_set_fixed = None
        self.model.fit(
            X=X,
            y=y,
            num_boost_round=self.num_boost_round,
            validation_data=val_set_fixed,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose,
            persist_train=False,
            time_bins=self.time_ruler
        )
        
    def score(self, X_num: ndarray, X_cat: ndarray | None, y: recarray) -> Tuple[float, float, float]:
        X = np.concatenate((X_num, X_cat), axis=-1) if X_cat is not None else X_num
        sf = self.model.predict(X).to_numpy()
        return score(self.y_train, sf, self.time_ruler, y)
