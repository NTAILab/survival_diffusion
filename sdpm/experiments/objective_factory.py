import numpy as np
from functools import partial
from sklearn.model_selection import StratifiedKFold, train_test_split

from ..loader import load_model_from_config
from ..naming import Metric, Model

import torch
import gc

NUM_BOOST_ROUND = 1000


def k_fold_wrapper(objective, K_stop, K_all):
    def wrapper(trial, X_num, X_cat, y, device, metric, seed):
        skf = StratifiedKFold(
            n_splits=K_all,
            shuffle=True,
            random_state=seed,
        )
        scores = []
        for _, (idx_train, idx_test) in zip(range(K_stop), skf.split(X_num, y["event"])):
            data = {
                "X_num_train": X_num[idx_train],
                "X_cat_train": X_cat[idx_train],
                "X_num_test": X_num[idx_test],
                "X_cat_test": X_cat[idx_test],
                "y_train": y[idx_train],
                "y_test": y[idx_test],
            }
            scores.append(objective(trial=trial, data=data, device=device, metric=metric, seed=seed))
            gc.collect()
            torch.cuda.empty_cache()
        return np.mean(scores).item()
    return wrapper

def sample_deepsurv_params(trial):
    params = {}
    params["n_hid_layers"] = trial.suggest_int("n_hid_layers", 1, 4)
    params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    params["activation"] = trial.suggest_categorical("activation", ["ReLU", "SiLU", "Tanh"])
    params["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
    params["norm"] = trial.suggest_categorical("norm", [False, True])
    params["dropout_mode"] = trial.suggest_categorical("dropout_mode", ["zero", "nonzero"])
    if params["dropout_mode"] == "zero":
        params["dropout"] = 0.0
    else:
        params["dropout"] = trial.suggest_float("dropout", 1e-2, 2e-1, log=True)
    return params

def sample_deephit_params(trial):
    params = sample_deepsurv_params(trial)
    params["alpha"] = trial.suggest_float("alpha", 0.0, 1.0)
    params["sigma"] = trial.suggest_categorical("sigma", [1e-2, 1e-1, 0.5, 1.0, 2.0])
    return params

def sample_sdpm_mlp_params(trial, f_use_rtdl: bool):
    params = {}
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    params["lr"] = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    params["val_period"] = 1
    params["layer_n"] = trial.suggest_int("layer_n", 2, 5)
    params["hidden_dim"] = trial.suggest_categorical("hidden_dim", [64, 128, 256, 512])
    params["activation"] = trial.suggest_categorical("activation", ["ReLU", "SiLU"])
    params["normalization"] = trial.suggest_categorical("normalization", ["none", "layer", "conditional"])
    params["cat_processor"] = trial.suggest_categorical("cat_processor", ["embedding", "empty"])
    if params["cat_processor"] == "embedding":
        params["cat_emb_dim"] = trial.suggest_categorical("cat_emb_dim", [4, 8])
    
    params["dropout_mode"] = trial.suggest_categorical("dropout_mode", ["zero", "nonzero"])
    if params["dropout_mode"] == "zero":
        params["dropout"] = 0.0
    else:
        params["dropout"] = trial.suggest_float("dropout", 0.05, 0.25, log=False)

    params["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4])

    params["time_steps_n"] = trial.suggest_categorical("time_steps_n", [11, 21, 31])
    params["time_emb_dim"] = trial.suggest_categorical("time_emb_dim", [8, 16])
    params["time_wrapper"] = "sinusoidal"
    if f_use_rtdl:
        rtdl_kind = "fourier"
        params["f_embed_noise"] = trial.suggest_categorical("f_embed_noise", [False, True])
        params["n_frequencies"] = trial.suggest_categorical("n_frequencies", [8, 16, 32, 64])
        params["frequency_init_scale"] = trial.suggest_float("frequency_init_scale", 1e-2, 1.0, log=True)
    else:
        rtdl_kind = "empty"
        params["rtdl_kind"] = rtdl_kind

    params["cosine_s"] = trial.suggest_float("cosine_s", 1e-4, 1.5e-1, log=True)
    # params["delta_scale"] = trial.suggest_categorical("delta_scale", [1, 3, 10])
    # params["warmup"] = trial.suggest_categorical("warmup", [10, 50, 100])
    return params

def sample_sdpm_tabm_params(trial):
    params = {}
    params["batch_size"] = trial.suggest_categorical("batch_size", [32, 64, 128])
    params["lr"] = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    params["val_period"] = trial.suggest_categorical("val_period", [1, 5, 10, 20])
    params["n_blocks"] = trial.suggest_int("n_blocks", 2, 4)
    params["d_block"] = trial.suggest_categorical("d_block", [64, 128, 256, 512])
    params["k"] = trial.suggest_categorical("k", [8, 16, 32, 64])
    params["activation"] = trial.suggest_categorical("activation", ["ReLU", "SiLU"])
    params["dropout_mode"] = trial.suggest_categorical("dropout_mode", ["zero", "nonzero"])
    if params["dropout_mode"] == "zero":
        params["dropout"] = 0.0
    else:
        params["dropout"] = trial.suggest_float("dropout", 5e-2, 2e-1, log=True)

    params["cat_processor"] = trial.suggest_categorical("cat_processor", ["embedding", "empty"])
    if params["cat_processor"] == "embedding":
        params["cat_emb_dim"] = trial.suggest_categorical("cat_emb_dim", [2, 4, 8])

    params["time_steps_n"] = trial.suggest_categorical("time_steps_n", [11, 21, 31])
    params["time_emb_dim"] = trial.suggest_categorical("time_emb_dim", [4, 8, 16])
    params["time_wrapper"] = trial.suggest_categorical("time_wrapper", ["time_embedding", "sinusoidal"])
    params["f_embed_noise"] = trial.suggest_categorical("f_embed_noise", [False, True])
    if params["f_embed_noise"]:
        rtdl_kind = "periodic"
    else:
        rtdl_kind = trial.suggest_categorical("rtdl_kind", ["piecewise", "periodic"])
        params["rtdl_kind"] = rtdl_kind
    params["rtdl_emb_dim"] = trial.suggest_categorical("rtdl_emb_dim", [8, 12, 16, 32])

    if rtdl_kind == "piecewise":
        params["rtdl_n_bins"] = trial.suggest_categorical("rtdl_n_bins", [16, 32, 48, 64, 96, 128])
        params["rtdl_target_aware"] = trial.suggest_categorical("rtdl_target_aware", [False, True])
    else:
        params["n_frequencies"] = trial.suggest_categorical("n_frequencies", [16, 32, 64, 96])
        params["periodic_lite"] = trial.suggest_categorical("periodic_lite", [False, True])
        params["frequency_init_scale"] = trial.suggest_float("frequency_init_scale", 1e-2, 1.0, log=True)
        if params["periodic_lite"]:
            params["rtdl_activation"] = True
        else:
            params["rtdl_activation"] = trial.suggest_categorical("rtdl_activation", [True, False])

    params["cosine_s"] = trial.suggest_float("cosine_s", 1e-4, 5e-2, log=True)
    return params

def sample_rsf_params(trial):
    params = {}
    params["n_estimators"] = trial.suggest_categorical("n_estimators", [50, 100, 200, 400, 800])
    params["max_depth"] = trial.suggest_categorical("max_depth", [None, 4, 8, 12, 16])
    params["min_samples_split"] = trial.suggest_categorical("min_samples_split", [2, 4, 8, 16, 0.01, 0.02, 0.05])
    params["min_samples_leaf"] = trial.suggest_categorical("min_samples_leaf", [1, 2, 4, 8, 0.005, 0.01, 0.02])
    params["max_features"] = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    params["max_leaf_nodes"] = trial.suggest_categorical("max_leaf_nodes", [None, 32, 64, 128, 256])
    params["bootstrap"] = trial.suggest_categorical("bootstrap", [True, False])
    return params

def sample_gbm_params(trial):
    params = {}
    params["early_stopping_rounds"] = trial.suggest_categorical("early_stopping_rounds", [10, 25, 50])
    params["learning_rate"] = trial.suggest_float("learning_rate", 1e-3, 2e-1, log=True)
    params["max_depth"] = trial.suggest_categorical("max_depth", [2, 3, 4, 5, 6, 8])
    params["min_child_weight"] = trial.suggest_float("min_child_weight", 1e-1, 20.0, log=True)
    params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0)
    params["colsample_bytree"] = trial.suggest_float("colsample_bytree", 0.5, 1.0)
    params["l2_reg"] = trial.suggest_float("l2_reg", 1e-4, 10.0, log=True)
    params["l1_reg"] = trial.suggest_float("l1_reg", 1e-4, 10.0, log=True)
    params["aft_loss_distribution"] = trial.suggest_categorical("aft_loss_distribution", ["normal", "logistic", "extreme"])
    params["aft_loss_distribution_scale"] = trial.suggest_float("aft_loss_distribution_scale", 0.1, 10.0, log=True)
    return params

def sample_gbm_wb_params(trial):
    params = sample_gbm_params(trial)
    params["wb_penalty"] = trial.suggest_float("wb_penalty", 1e-2, 10.0, log=True)
    params["wb_l1_ratio"] = trial.suggest_float("wb_l1_ratio", 0.0, 1.0)
    return params

def sample_gbm_km_params(trial):
    params = sample_gbm_params(trial)
    params["n_neighbours"] = int(trial.suggest_float("n_neighbours", 10, 100, step=10))
    return params

def get_trial_params(model_name: Model, trial):
    match model_name:
        case "deepsurv":
            return sample_deepsurv_params(trial)
        case "deephit":
            return sample_deephit_params(trial)
        case "sdpm_mlp" | "sdpm_mlp_exp" | "sdpm_mlp_no_emb":
            return sample_sdpm_mlp_params(trial, f_use_rtdl=(model_name!="sdpm_mlp_no_emb"))
        case "sdpm_tabm" | "sdpm_tabm_exp":
            return sample_sdpm_tabm_params(trial)
        case "rsf":
            return sample_rsf_params(trial)
        case "gbm_wb":
            return sample_gbm_wb_params(trial)
        case "gbm_km":
            return sample_gbm_km_params(trial)
        case _:
            raise NotImplementedError(f"Unknown model: {model_name}")

def resolve_params(model_name: Model, params: dict):
    params = params.copy()
    match model_name:
        case "deepsurv" | "deephit":
            dropout_mode = params.pop("dropout_mode", None)
            if dropout_mode == "zero":
                params["dropout"] = 0.0
            return params
        case "sdpm_mlp" | "sdpm_mlp_exp" | "sdpm_mlp_no_emb":
            if "val_period" not in params:
                params["val_period"] = 1
            dropout_mode = params.pop("dropout_mode", None)
            if dropout_mode == "zero":
                params["dropout"] = 0.0

            if model_name == "sdpm_mlp_no_emb":
                params["rtdl_kind"] = "empty"
            else:
                params["rtdl_kind"] = "fourier"

            time_steps_n = params.pop("time_steps_n")
            cosine_s = params.pop("cosine_s")
            params["scheduler"] = {
                "type": "cosine",
                "time_steps_n": time_steps_n,
                "s": cosine_s,
            }

            time_wrapper_type = params.pop("time_wrapper", "sinusoidal")
            time_emb_dim = params.pop("time_emb_dim")
            if time_wrapper_type == "time_embedding":
                params["time_wrapper"] = {
                    "type": "time_embedding",
                    "time_steps_n": time_steps_n,
                    "emb_dim": time_emb_dim,
                }
            else:
                params["time_wrapper"] = {
                    "type": "sinusoidal",
                    "emb_dim": time_emb_dim,
                }
            return params
        case "sdpm_tabm" | "sdpm_tabm_exp" | "sdpm_tabm_no_emb":
            dropout_mode = params.pop("dropout_mode", None)
            if dropout_mode == "zero":
                params["dropout"] = 0.0

            if model_name == "sdpm_tabm_no_emb":
                params["rtdl_kind"] = "empty"
            elif "rtdl_kind" not in params:
                params["rtdl_kind"] = "fourier"

            time_steps_n = params.pop("time_steps_n")
            cosine_s = params.pop("cosine_s")
            params["scheduler"] = {
                "type": "cosine",
                "time_steps_n": time_steps_n,
                "s": cosine_s,
            }

            time_wrapper_type = params.pop("time_wrapper")
            time_emb_dim = params.pop("time_emb_dim")
            if time_wrapper_type == "time_embedding":
                params["time_wrapper"] = {
                    "type": "time_embedding",
                    "time_steps_n": time_steps_n,
                    "emb_dim": time_emb_dim,
                }
            else:
                params["time_wrapper"] = {
                    "type": "sinusoidal",
                    "emb_dim": time_emb_dim,
                }
            return params
        case "rsf":
            return params
        case "gbm_wb":
            params["num_boost_round"] = NUM_BOOST_ROUND
            return params
        case "gbm_km":
            params["n_neighbours"] = int(params["n_neighbours"])
            params["num_boost_round"] = NUM_BOOST_ROUND
            return params
        case _:
            raise NotImplementedError(f"Unknown model: {model_name}")

def evaluate_model(model_name: Model, trial, data, device, metric, seed):
    x_num_train = data["X_num_train"]
    x_cat_train = data["X_cat_train"]
    y_train = data["y_train"]
    x_num_test = data["X_num_test"]
    x_cat_test = data["X_cat_test"]
    y_test = data["y_test"]

    model_params = resolve_params(model_name, get_trial_params(model_name, trial))
    model = load_model_from_config(
        model_name=model_name,
        config=model_params,
        device=device,
        X_num_train=x_num_train,
        X_cat_train=x_cat_train,
        y_train=y_train,
        seed=seed,
    )

    train_idx, val_idx = train_test_split(
        np.arange(y_train.shape[0]),
        test_size=0.25,
        stratify=y_train["event"],
        random_state=seed,
    )

    x_num_ptrain, x_num_pval = x_num_train[train_idx], x_num_train[val_idx]
    x_cat_ptrain, x_cat_pval = x_cat_train[train_idx], x_cat_train[val_idx]
    y_ptrain, y_pval = y_train[train_idx], y_train[val_idx]

    model.fit(
        x_num_ptrain,
        x_cat_ptrain,
        y_ptrain,
        (x_num_pval, x_cat_pval, y_pval),
    )
    score_kw = {
                'X_num': x_num_test,
                'X_cat': x_cat_test,
                'y': y_test
            }
    if model_name.startswith("sdpm"):
        score_kw['times_n'] = 512
    c_index, ibs, auc = model.score(**score_kw)
    scores = {
        "c_index": c_index,
        "ibs": ibs,
        "auc": auc,
    }
    return scores[metric]

def get_objective(
    X_num: np.ndarray,
    X_cat: np.ndarray,
    y: np.recarray,
    K_all: int,
    K_stop: int,
    model_name: Model,
    device: str,
    metric: Metric,
    seed: int,
):
    assert 1 <= K_stop <= K_all, f"K_stop must be in range [1, {K_all}]"
    wrap_objective = lambda obj: partial(
        k_fold_wrapper(obj, K_stop, K_all),
        X_num=X_num,
        X_cat=X_cat,
        y=y,
        metric=metric,
        device=device,
        seed=seed,
    )
    objective = partial(evaluate_model, model_name=model_name)
    return wrap_objective(objective)
