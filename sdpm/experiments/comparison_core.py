import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold

from ..logs import get_logger
from ..naming import Model, get_models_list
from .comparison import Result, set_seed, parse_repeats_spec, SEED_MAX, TEST_FOLDS, fit_and_score
from .util import load_data

def prepare_params_table(
        df: pd.DataFrame,
        model_name: Model,
        outer_seed: int,
        dataset: str,
    ) -> pd.DataFrame:
    required_cols = {
        "model_name",
        "repeat_i",
        "repeat_seed",
        "fold_i",
        "fold_seed",
        "params",
        "ds_name",
        "outer_seed",
    }
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Params table is missing columns: {sorted(missing_cols)}")

    work = df[
        (df["model_name"] == model_name)
        & (df["outer_seed"] == outer_seed)
        & (df["ds_name"] == dataset)
    ].copy()

    key_cols = [
        "model_name",
        "ds_name",
        "outer_seed",
        "repeat_i",
        "repeat_seed",
        "fold_i",
        "fold_seed",
    ]
    return work.drop_duplicates(subset=key_cols, keep="last")


def get_params(
        params_df: pd.DataFrame,
        *,
        model_name: Model,
        ds_name: str,
        outer_seed: int,
        repeat_i: int,
        repeat_seed: int,
        fold_i: int,
        fold_seed: int,
    ) -> str:
    matches = params_df[
        (params_df["model_name"] == model_name)
        & (params_df["ds_name"] == ds_name)
        & (params_df["outer_seed"] == outer_seed)
        & (params_df["repeat_i"] == repeat_i)
        & (params_df["repeat_seed"] == repeat_seed)
        & (params_df["fold_i"] == fold_i)
        & (params_df["fold_seed"] == fold_seed)
    ]
    if matches.empty:
        raise ValueError(
            "No params found for "
            f"model={model_name}, dataset={ds_name}, outer_seed={outer_seed}, "
            f"repeat_i={repeat_i}, repeat_seed={repeat_seed}, "
            f"fold_i={fold_i}, fold_seed={fold_seed}."
        )

    params = matches.iloc[-1]["params"]
    if pd.isna(params):
        raise ValueError(
            "Params are empty for "
            f"model={model_name}, dataset={ds_name}, repeat_i={repeat_i}, fold_i={fold_i}."
        )
    return str(params)


def do_test_core(
        model_name: Model,
        device: str,
        ds_name: str,
        params_df: pd.DataFrame,
        table_filename: str,
        repeats: int,
        seed: int,
        logger_filename: str | None = None,
        repeats_spec: set[int] | None = None,
    ):
    set_seed(seed)
    rng = np.random.default_rng(seed)
    logger = get_logger(f"Comparison core {ds_name}:{model_name}", logger_filename)

    data = load_data(ds_name)
    X_num = data["X_num"]
    X_cat = data["X_cat"]
    y = data["y"]

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
            random_state=seeds_repeats[r],
        )
        logger.info(f"Starting repeat {repeat_ord}/{repeats}")
        scores = {
            "c_index": [],
            "ibs": [],
            "auc": [],
            "time": [],
        }
        for i, (train_idx, test_idx) in enumerate(skf.split(X_num, y["event"])):
            X_num_train, X_cat_train, y_train = X_num[train_idx], X_cat[train_idx], y[train_idx]
            X_num_test, X_cat_test, y_test = X_num[test_idx], X_cat[test_idx], y[test_idx]
            
            params_str = get_params(
                params_df,
                model_name=model_name,
                ds_name=ds_name,
                outer_seed=seed,
                repeat_i=r,
                repeat_seed=seeds_repeats[r],
                fold_i=i,
                fold_seed=seeds_folds[i],
            )
            c_index, ibs, auc, wall_time = fit_and_score(
                model_name=model_name,
                device=device,
                X_num_train=X_num_train,
                X_cat_train=X_cat_train,
                y_train=y_train,
                X_num_test=X_num_test,
                X_cat_test=X_cat_test,
                y_test=y_test,
                params=json.loads(params_str),
                fold_seed=seeds_folds[i],
            )
            scores["c_index"].append(c_index)
            scores["ibs"].append(ibs)
            scores["auc"].append(auc)
            scores["time"].append(wall_time)

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
                params=params_str,
                ds_name=ds_name,
                outer_seed=seed,
            )
            with open(table_filename, "a") as file:
                file.write(result.get_row())

        c_index_mean = np.mean(scores["c_index"]).item()
        ibs_mean = np.mean(scores["ibs"]).item()
        auc_mean = np.mean(scores["auc"]).item()
        logger.info(f"Repeat {repeat_ord} results | C-index {c_index_mean:.3f} | IBS {ibs_mean:.3f} | AUC {auc_mean:.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-params_table", type=str, required=True)
    parser.add_argument("-data", type=str, required=True)
    parser.add_argument("-model", type=str, required=True, choices=get_models_list())
    parser.add_argument("-repeats", type=int, default=10)
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-device", type=str, default="cpu")
    parser.add_argument("-threads", type=int, default=8)
    parser.add_argument("-repeats_spec", type=str, default=None, nargs="+")
    parser.add_argument("-table_filename", type=str, default=None)
    parser.add_argument("-logger_filename", type=str, default=None)
    args = parser.parse_args()

    repeats_spec = parse_repeats_spec(args.repeats_spec, args.repeats)

    torch.set_num_threads(args.threads)

    params_df = pd.read_csv(args.params_table, sep=";")
    params_df = prepare_params_table(
        df=params_df,
        model_name=args.model,
        outer_seed=args.seed,
        dataset=args.data,
    )

    if args.table_filename is None:
        table_filename = "sdpm/experiments/results/comparison_core_table.csv"
    else:
        table_filename = args.table_filename
    table_path = Path(table_filename)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    if not table_path.exists() or table_path.stat().st_size == 0:
        with open(table_path, "w") as file:
            file.write(Result.get_header())
    
    do_test_core(
        model_name=args.model,
        device=args.device,
        ds_name=args.data,
        params_df=params_df,
        table_filename=table_filename,
        repeats=args.repeats,
        seed=args.seed,
        logger_filename=args.logger_filename,
        repeats_spec=repeats_spec,
    )


if __name__ == "__main__":
    main()
