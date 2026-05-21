from typing import Literal, Tuple
import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from sksurv.linear_model import CoxnetSurvivalAnalysis
from ...util import score
from .base import CompetitiveSksurvBase

class RSF(CompetitiveSksurvBase):
    def __init__(
        self,
        n_estimators: int,
        max_depth: int | None,
        min_samples_split: int | float,
        min_samples_leaf: int | float,
        max_features: Literal["sqrt", "log2"],
        max_leaf_nodes: int | None,
        bootstrap: bool,
        time_ruler: np.ndarray,
        n_jobs: int = 8,
        seed: int | None = None,
    ):
        super().__init__(time_ruler, seed=seed)
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            random_state=seed,
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = seed

    @staticmethod
    def _combine_features(
        X_num: np.ndarray | None,
        X_cat: np.ndarray | None,
    ) -> np.ndarray:
        if X_num is None and X_cat is None:
            raise ValueError("At least one of X_num or X_cat must be provided.")
        if X_num is None:
            return X_cat
        if X_cat is None:
            return X_num
        return np.concatenate((X_num, X_cat), axis=-1)

    def fit(
        self,
        X_num: np.ndarray | None,
        X_cat: np.ndarray | None,
        y: np.recarray,
        val_set: Tuple[np.ndarray | None, np.ndarray | None, np.recarray] | None = None,
    ):
        X_num_train = X_num
        X_cat_train = X_cat
        X_train = self._combine_features(X_num_train, X_cat_train)
        self.y_train = y.copy()
        self.model.fit(X_train, y)
        
    def predict(
        self,
        X_num: np.ndarray | None,
        X_cat: np.ndarray | None):
        X_test = self._combine_features(X_num, X_cat)
        sf_list = self.model.predict_survival_function(X_test, return_array=False)
        sf = np.empty((X_test.shape[0], self.time_ruler.shape[0]))
        domain = sf_list[0].domain
        times = self.time_ruler.clip(min=domain[0], max=domain[1] - 1e-6)
        for i, step_func in enumerate(sf_list):
            sf[i, :] = step_func(times)
        return sf

    def score(
        self,
        X_num: np.ndarray | None,
        X_cat: np.ndarray | None,
        y: np.recarray,
    ):
        sf = self.predict(X_num, X_cat)
        return score(self.y_train, sf, self.time_ruler, y)

    def predict_proba(self, X_num: np.ndarray | None, X_cat: np.ndarray | None):
        X = self._combine_features(X_num, X_cat)
        sf_list = self.model.predict_survival_function(X, False)
        sf = np.empty((X.shape[0], self.time_ruler.shape[0]))
        for i, step_func in enumerate(sf_list):
            sf[i, :] = step_func(self.time_ruler)
        proba = np.empty_like(sf)
        proba[:, :-1] = sf[:, :-1] - sf[:, 1:]
        proba[:, -1] = sf[:, -1]
        return proba
