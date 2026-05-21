import json
import numpy as np
from functools import cache
from typing import Dict, List, Iterable, Sequence
from ..util import delete_const_cats
from ..naming import (
    Model,
    Metric,
    get_ds_name,
    get_metric_name,
    get_model_name,
)
import pandas as pd
import scikit_posthocs as sp
import matplotlib.pyplot as plt

RC_PARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
    "font.size": 10,
    "legend.fontsize": 8,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.titlepad": 6,
    "mathtext.fontset": "dejavuserif",
    "savefig.dpi": 300,
}

@cache
def load_data(ds_name: str) -> Dict[str, np.ndarray]:
    path = f"data/{ds_name}.npz"
    with np.load(path) as data_file:
        data = dict(data_file)

        if "X_cat_train" in data and data["X_cat_train"].shape[1] > 0:
            x_cat_train, cat_mask = delete_const_cats(data["X_cat_train"])
            x_cat_val = delete_const_cats(data["X_cat_val"], cat_mask)
            x_cat_test = delete_const_cats(data["X_cat_test"], cat_mask)
        else:
            x_cat_train = None
            x_cat_val = None
            x_cat_test = None
        
        data.update(
            [
                ("X_cat_train", x_cat_train),
                ("X_cat_val", x_cat_val),
                ("X_cat_test", x_cat_test),
                
            ]
        )
        return data

    raise RuntimeError(f"Unable to load data '{path}'")

def load_configs(ds_name: str, models: Model | List[Model],
                metric_name: Metric = 'c_index') -> Dict:
    configs = {}
    path = f'sdpm/params_selection/params/{ds_name}/{metric_name}/'
    f_alone = False
    if not isinstance(models, list):
        models = [models]
        f_alone = True
    for model in models:
        file = open(path + f'{model}.json', 'r')
        configs[model] = json.load(file)
        file.close()
    if f_alone:
        configs = configs[models[0]]
    return configs

def delete_duplicates(df: pd.DataFrame, dedup_key: tuple[str, ...] = ("model_name", "ds_name", "fold_seed", "repeat_seed")):
    return df.drop_duplicates(subset=dedup_key, keep="last", inplace=False)

def summarize_experiments(
        df: pd.DataFrame,
        dedup_key: tuple[str, ...] = ("model_name", "ds_name", "fold_seed", "repeat_seed"),
        metric_cols: tuple[str, ...] = ("c_index", "ibs", "auc"),
        time_col: str = "time",
        repeat_col: str = "repeat_i",
        extra_group_cols: tuple[str, ...] = tuple(),
    ) -> pd.DataFrame:

    df = delete_duplicates(df, dedup_key)

    group_cols = [*extra_group_cols, "model_name", "ds_name", repeat_col]

    agg_dict = {col: "mean" for col in metric_cols}
    agg_dict[time_col] = "mean"

    result = (
        df.groupby(group_cols, as_index=False)
          .agg(agg_dict)
          .rename(columns={time_col: f"mean_{time_col}"})
          .sort_values(group_cols)
          .reset_index(drop=True)
    )
    return result

def aggregate_over_repeats(
        df: pd.DataFrame,
        metric_cols=("c_index", "ibs", "auc"),
        time_col="mean_time",
    ):
    group_cols = ["ds_name", "model_name"]

    agg_dict = {col: "mean" for col in metric_cols}
    agg_dict[time_col] = "mean"

    result = (
        df.groupby(group_cols, as_index=False)
          .agg(agg_dict)
          .sort_values(group_cols)
          .reset_index(drop=True)
    )
    return result

def ranks_from_results(
        df: pd.DataFrame,
        models: Sequence[str],
        metric_cols: Sequence[str] = ("c_index", "ibs", "auc"),
        bigger_is_better: dict[str, bool] | None = None,
        group_cols: tuple[str, ...] = ("ds_name", "fold_seed"),
        model_col: str = "model_name",
        dedup_key: tuple[str, ...] = ("model_name", "ds_name", "fold_seed", "repeat_seed"),
    ) -> pd.DataFrame:

    required_cols = {model_col, *group_cols, *metric_cols}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataframe is missing columns: {sorted(missing_cols)}")

    if bigger_is_better is None:
        bigger_is_better = {
            "c_index": True,
            "ibs": False,
            "auc": True,
        }

    rank_cols = [f"{metric}_rank" for metric in metric_cols]
    work = delete_duplicates(df, dedup_key)
    group_df = work[list(group_cols)].drop_duplicates()
    work = work[work[model_col].isin(models)].copy()

    index_cols = [*group_cols, model_col]
    rank_df = (
        work[index_cols]
        .drop_duplicates()
        .set_index(index_cols)
    )

    for metric in metric_cols:
        ascending = not bigger_is_better.get(metric, True)
        metric_ranks = (
            work.groupby(list(group_cols))[metric]
            .rank(method="average", ascending=ascending, na_option="keep")
        )
        metric_rank_df = work[index_cols].copy()
        metric_rank_df[f"{metric}_rank"] = metric_ranks
        metric_rank_df = (
            metric_rank_df.groupby(index_cols, as_index=True)[f"{metric}_rank"]
            .last()
            .to_frame()
        )
        rank_df = rank_df.join(metric_rank_df, how="left")

    full_index_df = group_df.merge(
        pd.DataFrame({model_col: list(models)}),
        how="cross",
    )
    full_index = pd.MultiIndex.from_frame(full_index_df[index_cols])
    result = (
        rank_df.reindex(full_index)
        .reset_index()
        .sort_values(index_cols)
        .reset_index(drop=True)
    )
    return result[[*index_cols, *rank_cols]]

def check_results(
        df: pd.DataFrame,
        folds: int,
        repeats: int,
        models: Sequence[str] | None = None,
        outer_seed: int = 42,
        model_col: str = "model_name",
        dataset_col: str = "ds_name",
        repeat_col: str = "repeat_i",
        fold_col: str = "fold_i",
        outer_seed_col: str = "outer_seed",
    ) -> bool:

    required_cols = {model_col, dataset_col, repeat_col, fold_col, outer_seed_col}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(f"Dataframe is missing columns: {sorted(missing_cols)}")
    if folds <= 0:
        raise ValueError("folds must be positive")
    if repeats <= 0:
        raise ValueError("repeats must be positive")

    expected_repeats = set(range(repeats))
    expected_folds = set(range(folds))

    work = df[df[outer_seed_col] == outer_seed].copy()
    if models is None:
        models_to_check = sorted(df[model_col].dropna().unique().tolist())
    else:
        models_to_check = list(models)

    datasets_to_check = sorted(df[dataset_col].dropna().unique().tolist())
    is_ok = True

    for model in models_to_check:
        for dataset in datasets_to_check:
            pair_df = work[
                (work[model_col] == model)
                & (work[dataset_col] == dataset)
            ]
            present_repeats = set(pair_df[repeat_col].dropna().astype(int).unique())
            missing_repeats = expected_repeats - present_repeats
            extra_repeats = present_repeats - expected_repeats

            incomplete_folds = {}
            for repeat_i in sorted(expected_repeats & present_repeats):
                repeat_df = pair_df[pair_df[repeat_col] == repeat_i]
                present_folds = set(repeat_df[fold_col].dropna().astype(int).unique())
                missing_folds = expected_folds - present_folds
                extra_folds = present_folds - expected_folds
                if missing_folds or extra_folds:
                    incomplete_folds[repeat_i] = (present_folds, missing_folds, extra_folds)

            if missing_repeats or extra_repeats or incomplete_folds:
                is_ok = False
                repeats_msg = _format_int_values(present_repeats)
                print(f"[ERROR] Model `{model}`, dataset `{dataset}` has only repeats {repeats_msg}")

                if missing_repeats:
                    print(f"        Missing repeats: {_format_int_values(missing_repeats)}")
                if extra_repeats:
                    print(f"        Unexpected repeats: {_format_int_values(extra_repeats)}")

                for repeat_i, (present_folds, missing_folds, extra_folds) in incomplete_folds.items():
                    print(
                        f"        Repeat {repeat_i} has only folds "
                        f"{_format_int_values(present_folds)}"
                    )
                    if missing_folds:
                        print(f"        Missing folds: {_format_int_values(missing_folds)}")
                    if extra_folds:
                        print(f"        Unexpected folds: {_format_int_values(extra_folds)}")
            else:
                print(f"[OK] Model `{model}`, dataset `{dataset}` is ok")

    return is_ok

def _format_int_values(values: Iterable[int]) -> str:
    sorted_values = sorted(values)
    if not sorted_values:
        return "none"
    return ", ".join(str(value) for value in sorted_values)

def metric_matrix_from_melt(
        df: pd.DataFrame,
        metric: str,
        models: Sequence[str] | None = None,
        datasets: Sequence[str] | None = None,
        model_col: str = "model_name",
        dataset_col: str = "ds_name",
        repeat_col: str = "repeat_i",
    ) -> pd.DataFrame:

    required_cols = {model_col, dataset_col, repeat_col, metric}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В датафрейме отсутствуют колонки: {sorted(missing)}")

    work = df.copy()

    if models is not None:
        work = work[work[model_col].isin(models)]

    if datasets is not None:
        work = work[work[dataset_col].isin(datasets)]

    avg_df = (
        work.groupby([model_col, dataset_col], as_index=False)[metric]
        .mean()
    )

    result = avg_df.pivot(
        index=model_col,
        columns=dataset_col,
        values=metric,
    )

    if models is not None:
        result = result.reindex(models)

    if datasets is not None:
        result = result.reindex(columns=datasets)

    return result

def metric_latex_table_from_melt(
        df: pd.DataFrame,
        metric: Metric,
        models: Sequence[str] | None = None,
        datasets: Sequence[str] | None = None,
        model_col: str = "model_name",
        dataset_col: str = "ds_name",
        repeat_col: str = "repeat_i",
        precision: int = 3,
        na_rep: str = "",
        caption: str | None = None,
        label: str | None = None,
    ) -> str:

    required_cols = {model_col, dataset_col, repeat_col, metric}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"В датафрейме отсутствуют колонки: {sorted(missing)}")

    bigger_is_better = {
        "c_index": True,
        "auc": True,
        "ibs": False,
    }
    if metric not in bigger_is_better:
        raise ValueError(f"Неизвестное направление оптимизации для метрики: {metric}")

    work = df.copy()

    if models is not None:
        work = work[work[model_col].isin(models)]

    if datasets is not None:
        work = work[work[dataset_col].isin(datasets)]

    stats_df = (
        work.groupby([model_col, dataset_col], as_index=False)
        .agg(
            mean=(metric, "mean"),
            std=(metric, "std"),
        )
    )
    best_agg = "max" if bigger_is_better[metric] else "min"
    best_means = stats_df.groupby(dataset_col)["mean"].transform(best_agg)
    stats_df["is_best"] = stats_df["mean"].eq(best_means) & stats_df["mean"].notna()

    def format_cell(row: pd.Series) -> str:
        mean = row["mean"]
        std = row["std"]
        if pd.isna(mean):
            return na_rep
        value = f"{mean:.{precision}f}"
        if pd.isna(std):
            cell = value
        else:
            cell = f"{value} {{\\scriptstyle \\pm {std:.{precision}f}}}"
        if row["is_best"]:
            return f"$\\mathbf{{{cell}}}$"
        return f"${cell}$"

    stats_df["value"] = stats_df.apply(format_cell, axis=1)

    result = stats_df.pivot(
        index=model_col,
        columns=dataset_col,
        values="value",
    )

    if models is not None:
        result = result.reindex(models)

    if datasets is not None:
        result = result.reindex(columns=datasets)

    result = result.rename(
        index={model: get_model_name(model, f_latex=True) for model in result.index},
        columns={dataset: get_ds_name(dataset) for dataset in result.columns},
    )
    result.index.name = None
    result.columns.name = None

    latex_df = result.T
    column_format = "l" + "".join(["c"] * (len(latex_df.columns)))

    latex = latex_df.to_latex(
        escape=False,
        na_rep=na_rep,
        caption=caption,
        label=label,
        column_format=column_format,
    )
    return latex

def plot_cd_diagram_paper(
    ranks,
    sig_matrix,
    *,
    alpha=0.05,
    figsize=(7, 2.2),
    left_only=False,
    text_h_margin=0.05
):
    with plt.rc_context(RC_PARAMS):
        fig, ax = plt.subplots(figsize=figsize)

        sp.critical_difference_diagram(
            ranks=ranks,
            sig_matrix=sig_matrix,
            alpha=alpha,
            ax=ax,

            label_fmt_left="{label} ",
            label_fmt_right=" {label}",

            label_props={
                "fontsize": 10,
                "fontfamily": "serif",
            },

            marker_props={
                "marker": 2,
                "s": 80,
                "color": "black",
                "zorder": 3,
            },

            elbow_props={
                "color": "black",
                "linewidth": 0.8,
            },

            crossbar_props={
                "color": "red",
                "linewidth": 1.6,
                "dash_joinstyle": "round",
                "solid_joinstyle": "round",
                "solid_capstyle": "round",
                "antialiased": True,
            },

            text_h_margin=text_h_margin,
            left_only=left_only,
        )

        plt.tight_layout()
        return fig, ax

def make_relative_improvement_df(
    df: pd.DataFrame,
    baseline_model: str,
    metric: str,
    models: list[str] | None = None,
    higher_is_better: bool = True,
    split_keys: list[str] | None = None,
) -> pd.DataFrame:

    if split_keys is None:
        split_keys = [
            "ds_name",
            "outer_seed",
            "repeat_i",
            "repeat_seed",
            "fold_i",
            "fold_seed"
        ]

    if models is None:
        models = sorted(df["model_name"].unique().tolist())
        models = [m for m in models if m != baseline_model]

    work_models = models + [baseline_model]

    sub = df[df["model_name"].isin(work_models)].copy()

    base = (
        sub[sub["model_name"] == baseline_model]
        [split_keys + [metric]]
        .rename(columns={metric: "baseline_value"})
    )

    merged = sub.merge(base, on=split_keys, how="inner")

    merged = merged[merged["model_name"] != baseline_model].copy()

    if higher_is_better:
        merged["relative_value"] = merged[metric] / merged["baseline_value"]
    else:
        merged["relative_value"] = merged["baseline_value"] / merged[metric]

    result = merged[
        split_keys + [
            "ds_name",
            "model_name",
            metric,
            "baseline_value",
            "relative_value"
        ]
    ].copy()

    result["metric"] = metric
    result = result.rename(columns={metric: "raw_value"})

    return result.reset_index(drop=True)
