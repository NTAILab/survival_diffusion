import json
from pathlib import Path
from .baseline.mlp import MLPBaseline
from .diffusion.process import DDPM
from .experiments.models.py_cox import DeepHit, DeepSurv
from .experiments.models.sk_surv import RSF
from .experiments.models.xg_boost import AFTModel, GBMKM, GBMWeibull
from .feat_processing import (
    DefaultFeatProcessor,
    EmptyNumericProcessor,
    EmbeddingCatProcessor,
    EmptyCatProcessor,
    FourierFeatProcessor
)
from .sdpm import SDPM
import torch
import numpy as np
from .naming import Model
from typing import Dict

def load_model_from_config(
    model_name: Model,
    config: Dict,
    *,
    device: torch.device | str = "cpu",
    X_num_train: np.ndarray | None = None,
    X_cat_train: np.ndarray | None = None,
    y_train: np.recarray | None = None,
    verbose: int = 0,
    seed: int | None = None
):

    if model_name == "rsf":
        if y_train is None:
            raise ValueError("y_train is required to build RSF model from config.")
        return RSF(
            n_estimators=config["n_estimators"],
            max_depth=config["max_depth"],
            min_samples_split=config["min_samples_split"],
            min_samples_leaf=config["min_samples_leaf"],
            max_features=config["max_features"],
            max_leaf_nodes=config["max_leaf_nodes"],
            bootstrap=config["bootstrap"],
            time_ruler=np.unique(y_train["time"]),
            n_jobs=8,
            seed=seed,
        )

    if X_num_train is None:
        raise ValueError(f"X_num_train is required to build {model_name} model from config.")

    if model_name == "deepsurv":
        if X_cat_train is None:
            x_cat_dim = 0
        else:
            x_cat_dim = X_cat_train.shape[1]
        return DeepSurv(
            device=device,
            x_num_dim=X_num_train.shape[1],
            x_cat_dim=x_cat_dim,
            n_hid_layers=config["n_hid_layers"],
            hidden_dim=config["hidden_dim"],
            norm=config["norm"],
            activation=config["activation"],
            dropout=config["dropout"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            epochs=1000,
            verbose=bool(verbose),
            seed=seed
        )

    if model_name == "aft":
        if y_train is None:
            raise ValueError("y_train is required to build AFT model from config.")
        return AFTModel(
            num_boost_round=config["num_boost_round"],
            time_ruler=np.unique(y_train["time"]),
            early_stopping_rounds=config["early_stopping_rounds"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            min_child_weight=config.get("min_child_weight", 1.0),
            subsample=config.get("subsample", 1.0),
            colsample_bytree=config.get("colsample_bytree", 1.0),
            gamma=config.get("gamma", 0.0),
            l2_reg=config["l2_reg"],
            l1_reg=config["l1_reg"],
            tree_method=config.get("tree_method", "hist"),
            max_bin=config.get("max_bin", 256),
            aft_loss_distribution=config["aft_loss_distribution"],
            aft_loss_distribution_scale=config["aft_loss_distribution_scale"],
            n_jobs=8,
            seed=seed,
            verbose=bool(verbose)
        )
        
    if model_name == 'gbm_km':
        if y_train is None:
            raise ValueError("y_train is required to build GBMKM model from config.")
        return GBMKM(
            num_boost_round=config["num_boost_round"],
            time_ruler=np.unique(y_train["time"]),
            early_stopping_rounds=config["early_stopping_rounds"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            min_child_weight=config.get("min_child_weight", 1.0),
            subsample=config.get("subsample", 1.0),
            colsample_bytree=config.get("colsample_bytree", 1.0),
            gamma=config.get("gamma", 0.0),
            l2_reg=config["l2_reg"],
            l1_reg=config["l1_reg"],
            tree_method=config.get("tree_method", "hist"),
            max_bin=config.get("max_bin", 256),
            aft_loss_distribution=config["aft_loss_distribution"],
            aft_loss_distribution_scale=config["aft_loss_distribution_scale"],
            n_neighbors=config["n_neighbours"],
            n_jobs=8,
            verbose=False,
            seed=seed
        )
    if model_name == 'gbm_wb':
        if y_train is None:
            raise ValueError("y_train is required to build GBMWeibull model from config.")
        return GBMWeibull(
            num_boost_round=config["num_boost_round"],
            time_ruler=np.unique(y_train["time"]),
            early_stopping_rounds=config["early_stopping_rounds"],
            learning_rate=config["learning_rate"],
            max_depth=config["max_depth"],
            min_child_weight=config.get("min_child_weight", 1.0),
            subsample=config.get("subsample", 1.0),
            colsample_bytree=config.get("colsample_bytree", 1.0),
            gamma=config.get("gamma", 0.0),
            l2_reg=config["l2_reg"],
            l1_reg=config["l1_reg"],
            tree_method=config.get("tree_method", "hist"),
            max_bin=config.get("max_bin", 256),
            aft_loss_distribution=config["aft_loss_distribution"],
            aft_loss_distribution_scale=config["aft_loss_distribution_scale"],
            wb_penalty=config.get("wb_penalty", 0.0),
            wb_l1_ratio=config["wb_l1_ratio"],
            n_jobs=8,
            verbose=False,
            seed=seed
        )

    if model_name == "deephit":
        if y_train is None:
            raise ValueError("y_train is required to build DeepHit model from config.")
        if X_cat_train is None:
            x_cat_dim = 0
        else:
            x_cat_dim = X_cat_train.shape[1]
        return DeepHit(
            device=device,
            x_num_dim=X_num_train.shape[1],
            x_cat_dim=x_cat_dim,
            time_ruler=np.unique(y_train["time"]),
            n_hid_layers=config["n_hid_layers"],
            hidden_dim=config["hidden_dim"],
            norm=config["norm"],
            activation=config["activation"],
            dropout=config["dropout"],
            lr=config["lr"],
            batch_size=config["batch_size"],
            epochs=1000,
            alpha=config["alpha"],
            sigma=config["sigma"],
            verbose=bool(verbose),
            seed=seed
        )

    f_embed_noise = config.get("f_embed_noise", False)
    rtdl_kind = config.get("rtdl_kind", "periodic")
    if f_embed_noise:
        if rtdl_kind not in ["periodic", "fourier"]:
            raise ValueError("Invalid config: f_embed_noise=True requires periodic embeddings.")

    n_num_features = X_num_train.shape[1] + (2 if f_embed_noise else 0)

    if rtdl_kind == "empty":
        num_proc = EmptyNumericProcessor(n_num_features)
    else:
        torch.manual_seed(seed)
        if rtdl_kind == "piecewise":
            raise NotImplementedError()
        elif rtdl_kind == "periodic":
            raise NotImplementedError()
        elif rtdl_kind == "fourier":
            num_proc = FourierFeatProcessor(
                n_features=n_num_features,
                n_frequencies=config["n_frequencies"],
                frequency_init_scale=config["frequency_init_scale"]
            )
        else:
            raise ValueError(f"Unknown rtdl_kind '{rtdl_kind}' in config.")

    n_cat_features = 0 if X_cat_train is None else X_cat_train.shape[1]
    cat_proc_type = config["cat_processor"]
    if cat_proc_type == "embedding":
        cat_emb_dim = config["cat_emb_dim"]
        cat_proc = EmbeddingCatProcessor(n_cat_features, cat_emb_dim)
    elif cat_proc_type == "empty":
        cat_proc = EmptyCatProcessor(n_cat_features)
    else:
        raise ValueError(f"Unknown cat_processor '{cat_proc_type}' in config.")

    feat_proc = DefaultFeatProcessor(num_proc, cat_proc)

    scheduler_config = config.get("scheduler")
    if scheduler_config is None:
        # Backward compatibility with legacy flat configs.
        time_steps_n = config["time_steps_n"]
        cosine_s = config.get("cosine_s", 0.008)
        scheduler_config = {"type": "cosine", "time_steps_n": time_steps_n, "s": cosine_s}
    scheduler = DDPM.get_scheduler(device, scheduler_config)

    time_wrapper_config = config.get("time_wrapper")
    if isinstance(time_wrapper_config, str):
        # Backward compatibility if only type string is present.
        time_steps_n = scheduler_config["time_steps_n"]
        if time_wrapper_config == "time_embedding":
            time_wrapper_config = {
                "type": "time_embedding",
                "time_steps_n": time_steps_n,
                "emb_dim": config["time_emb_dim"],
            }
        elif time_wrapper_config == "sinusoidal":
            time_wrapper_config = {"type": "sinusoidal", "emb_dim": config["time_emb_dim"]}
        else:
            raise ValueError(f"Unknown time_wrapper '{time_wrapper_config}' in config.")
    time_wrapper = DDPM.get_time_wrapper(device, time_wrapper_config, seed=seed)

    loss = torch.nn.MSELoss()

    diffusion = DDPM(
        device=device,
        scheduler=scheduler,
        time_wrapper=time_wrapper,
        feat_processor=feat_proc,
        f_embed_noise=f_embed_noise,
        loss=loss
    )

    if model_name.startswith("sdpm_mlp"):
        weight_decay = config.get("weight_decay", 0.0)
        normalization = config["normalization"]
        if isinstance(normalization, bool):
            normalization = 'layer' if normalization else 'none'
        baseline = MLPBaseline(
            layer_n=config["layer_n"],
            input_dim=diffusion.get_mlp_input_dim(),
            hidden_dim=config["hidden_dim"],
            activation=config["activation"],
            normalization=normalization,
            dropout=config.get("dropout", 0.0),
            seed=seed
        )
        f_log_time = "pure_time" not in model_name
        f_gauss_delta = "pure_delta" not in model_name
        if "pure_all" in model_name:
            f_log_time = f_gauss_delta = False
    else:
        raise ValueError(f"Unknown model_name '{model_name}'.")

    return SDPM(
        baseline=baseline,
        diffusion=diffusion,
        epochs_n=1000,
        val_period=config["val_period"],
        batch_size=config["batch_size"],
        warmup=config.get("warmup", None),
        optimizer="AdamW",
        f_log_time=f_log_time,
        f_gauss_delta=f_gauss_delta,
        delta_scale=config.get("delta_scale", 1.0),
        optimizer_params={"lr": config["lr"], "weight_decay": weight_decay},
        verbose=verbose,
        patience=config.get("patience", None),
        val_metric=config.get("val_metric", "c_index"),
        seed=seed
    )
