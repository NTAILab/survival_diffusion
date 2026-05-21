import json
import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from lifelines.exceptions import ConvergenceError
from ..naming import Model, Metric
from ..naming import get_models_list, get_metrics_list
from ..logs import get_logger
from .objective_factory import get_objective, resolve_params
import optuna
from ..loader import load_model_from_config
from dataclasses import dataclass, fields
import torch
import random
from sklearn.model_selection import train_test_split
from .util import load_data
import time
from typing import List
from inspect import signature
from lifelines.exceptions import ConvergenceError

SEED_MAX = 2 ** 20
TEST_FOLDS = 4
OPTUNA_FOLDS = 4
DATASET = ''
SEED = 42

@dataclass(frozen=True)
class Result:
    model_name: str
    c_index: float
    ibs: float
    auc: float
    time: float
    repeat_i: int
    repeat_seed: int
    fold_i: int
    fold_seed: int
    params: str
    ds_name: str = DATASET
    outer_seed: int = SEED
    
    @staticmethod
    def get_header() -> str:
        field_names = [field.name for field in fields(Result)]
        return ';'.join(field_names) + '\n'
    
    def get_row(self) -> str:
        get_val = lambda field: getattr(self, field.name)
        field_values = [str(get_val(field)) for field in fields(Result)]
        return ';'.join(field_values) + '\n'

def do_optuna_param_search(model_name: Model, metric: Metric,
                           X_num: np.ndarray, X_cat: np.ndarray, 
                           y: np.recarray, device: str,
                           trials: int, K: int, seed: int):
    study_direction = "minimize" if metric == "ibs" else "maximize"
    sampler = optuna.samplers.TPESampler(seed=seed)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction=study_direction, sampler=sampler)
    objective = get_objective(
        X_num=X_num,
        X_cat=X_cat,
        y=y,
        K_all=OPTUNA_FOLDS,
        K_stop=K,
        model_name=model_name,
        device=device,
        metric=metric,
        seed=seed
    )
    study.optimize(objective, n_trials=trials,
                   show_progress_bar=True, catch=[ConvergenceError])
    best_params = resolve_params(model_name, study.best_trial.params)
    return best_params

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)   


def parse_repeats_spec(spec: List[str] | None, repeats: int) -> set[int] | None:
    if spec is None:
        return None
    parts = [int(s) for s in spec]
    repeats_set = set()
    for part in parts:
        repeat_i = int(part)
        if repeat_i <= 0 or repeat_i > repeats:
            raise ValueError(f"repeats_spec must contain only integers in range [1, {repeats}]")
        repeats_set.add(repeat_i)
    return repeats_set

def fit_and_score(
        model_name: Model,
        device: str,
        X_num_train: np.ndarray,
        X_cat_train: np.ndarray,
        y_train: np.recarray,
        X_num_test: np.ndarray,
        X_cat_test: np.ndarray,
        y_test: np.recarray,
        params: dict,
        fold_seed: int,
    ) -> tuple[float, float, float, float]:

    model = load_model_from_config(
        model_name=model_name,
        config=params,
        device=device,
        X_num_train=X_num_train,
        X_cat_train=X_cat_train,
        y_train=y_train,
        seed=fold_seed,
    )
    train_idx, val_idx = train_test_split(
        np.arange(y_train.shape[0]),
        test_size=0.25,
        stratify=y_train["event"],
        random_state=fold_seed,
    )

    X_num_ptrain, X_num_pval = X_num_train[train_idx], X_num_train[val_idx]
    X_cat_ptrain, X_cat_pval = X_cat_train[train_idx], X_cat_train[val_idx]
    y_ptrain, y_pval = y_train[train_idx], y_train[val_idx]

    time_stamp = time.time()
    model.fit(
        X_num_ptrain,
        X_cat_ptrain,
        y_ptrain,
        (X_num_pval, X_cat_pval, y_pval),
    )
    wall_time = time.time() - time_stamp

    score_kw = {
        "X_num": X_num_test,
        "X_cat": X_cat_test,
        "y": y_test,
    }
    if model_name.startswith("sdpm"):
        score_kw["times_n"] = 2048
    c_index, ibs, auc = model.score(**score_kw)
    return c_index, ibs, auc, wall_time

def do_test(model_name: Model, device: str,
            X_num: np.ndarray, X_cat: np.ndarray,
            y: np.recarray, repeats: int,
            K: int, trials: int, val_metric: Metric,
            table_filename: str, seed: int,
            logger_filename: str | None = None,
            repeats_spec: set[int] | None = None):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    logger = get_logger(f"Comparison {DATASET}:{model_name}", logger_filename)
    seeds_repeats = rng.integers(1, SEED_MAX, repeats).tolist()
    logger.info(f"Repeats seeds: {seeds_repeats}")
    if repeats_spec is not None:
        logger.info(f"Selected repeats (1-based): {sorted(repeats_spec)}")
    for r in range(repeats):
        seeds_folds = rng.integers(1, SEED_MAX, TEST_FOLDS).tolist()
        repeat_ord = r + 1
        logger.info(f"Folds seeds: {seeds_folds}")
        if repeats_spec is not None and repeat_ord not in repeats_spec:
            logger.info(f"Skipping repeat {repeat_ord}/{repeats}")
            continue
        skf = StratifiedKFold(
            n_splits=TEST_FOLDS,
            shuffle=True,
            random_state=seeds_repeats[r]
        )
        logger.info(f"Starting repeat {repeat_ord}/{repeats}")
        scores = {
            'c_index': [],
            'ibs': [],
            'auc': [],
            'time': []
        }
        for i, (train_idx, test_idx) in enumerate(skf.split(X_num, y["event"])):
            X_num_train, X_cat_train, y_train = X_num[train_idx], X_cat[train_idx], y[train_idx]
            X_num_test, X_cat_test, y_test = X_num[test_idx], X_cat[test_idx], y[test_idx]
            params = do_optuna_param_search(
                model_name=model_name,
                metric=val_metric,
                X_num=X_num_train,
                X_cat=X_cat_train,
                y=y_train,
                device=device,
                trials=trials,
                K=K,
                seed=seeds_folds[i])
            
            try:
                c_index, ibs, auc, wall_time = fit_and_score(
                    model_name=model_name,
                    X_num_train=X_num_train,
                    X_cat_train=X_cat_train,
                    y_train=y_train,
                    X_num_test=X_num_test,
                    X_cat_test=X_cat_test,
                    y_test=y_test,
                    params=params,
                    fold_seed=seeds_folds[i],
                    device=device
                )
            except ConvergenceError:
                print("GBSEStackedWeibull convergence error")
                print("Dropping very large times")
                try:
                    good_idx = np.argwhere(y_train['time'] < np.percentile(y_train['time'], 0.95)).ravel()
                    c_index, ibs, auc, wall_time = fit_and_score(
                        model_name=model_name,
                        X_num_train=X_num_train[good_idx],
                        X_cat_train=X_cat_train[good_idx],
                        y_train=y_train[good_idx],
                        X_num_test=X_num_test,
                        X_cat_test=X_cat_test,
                        y_test=y_test,
                        params=params,
                        fold_seed=seeds_folds[i],
                        device=device
                    )
                    if np.isnan(ibs):
                        ibs = 1.0
                    if np.isnan(auc):
                        auc = 0.5
                except Exception:
                    print("Unable to fit GBSEStackedWeibull")
                    c_index, ibs, auc, wall_time = 0.5, 1.0, 0.5, 0

            scores['c_index'].append(c_index)
            scores['ibs'].append(ibs)
            scores['auc'].append(auc)
            scores['time'].append(wall_time)
            
            logger.info(f"Fold {i + 1}/{TEST_FOLDS} | C-index {c_index:.3f} | IBS {ibs:.3f} | AUC {auc:.3f}")
            result = Result(
                model_name=model_name,
                c_index=c_index,
                ibs=ibs,
                auc=auc,
                time=wall_time,
                repeat_i=r,
                repeat_seed=seeds_repeats[r],
                fold_i=i,
                fold_seed=seeds_folds[i],
                params=json.dumps(params),
                ds_name=DATASET,
                outer_seed=SEED,
            )
            if table_filename is not None:
                with open(table_filename, 'a') as file:
                    file.write(result.get_row())
        c_index_mean = np.mean(scores['c_index']).item()
        ibs_mean = np.mean(scores['ibs']).item()
        auc_mean = np.mean(scores['auc']).item()
        logger.info(f"Repeat {repeat_ord} results | C-index {c_index_mean:.3f} | IBS {ibs_mean:.3f} | AUC {auc_mean:.3f}")


def main():
    global DATASET, SEED

    parser = argparse.ArgumentParser()
    parser.add_argument("-data", type=str, required=True)
    parser.add_argument("-model", type=str, required=True, choices=get_models_list())
    parser.add_argument("-trials", type=int, default=100)
    parser.add_argument("-repeats", type=int, default=10)
    parser.add_argument("-k_optuna", type=int, default=1)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-device", type=str, default="cpu")
    parser.add_argument("-threads", type=int, default=8)
    parser.add_argument("-repeats_spec", type=str, default=None, nargs='+')
    parser.add_argument("-val_metric", type=str, default="c_index", choices=get_metrics_list())
    parser.add_argument("-table_filename", type=str, default=None)
    parser.add_argument("-logger_filename", type=str, default=None)
    args = parser.parse_args()

    repeats_spec = parse_repeats_spec(args.repeats_spec, args.repeats)

    torch.set_num_threads(args.threads)
    DATASET = args.data
    SEED = args.seed

    data = load_data(DATASET)
    X_num = data['X_num']
    X_cat = data['X_cat']
    y = data['y']

    if args.table_filename is None:
        table_filename = 'sdpm/experiments/results/comparison_table.csv'
    else:
        table_filename = args.table_filename
    table_path = Path(table_filename)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if not table_path.exists() or table_path.stat().st_size == 0:
        with open(table_path, "w") as file:
            file.write(Result.get_header())

    do_test(
        model_name=args.model,
        device=args.device,
        X_num=X_num,
        X_cat=X_cat,
        y=y,
        repeats=args.repeats,
        K=args.k_optuna,
        trials=args.trials,
        val_metric=args.val_metric,
        table_filename=table_filename,
        seed=args.seed,
        logger_filename=args.logger_filename,
        repeats_spec=repeats_spec,
    )
 

if __name__ == "__main__":
    main()
