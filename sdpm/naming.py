from typing import Literal, List, get_args

Model = Literal[
    "sdpm_mlp",
    "rsf",
    "deepsurv",
    "deephit",
    'gbm_wb',
    'gbm_km',
    # "sdpm_mlp_exp",
    # "sdpm_tabm_exp",
    # "sdpm_mlp_no_emb",
    # "sdpm_tabm_no_emb",
    # "sdpm_mlp_pure_time",
    # "sdpm_mlp_pure_delta",
    # "sdpm_mlp_pure_all",
]

Metric = Literal[
    "c_index",
    "ibs",
    "auc"
]

Dataset = Literal[
    'flchain',
    'seer',
    'support',
    'rotterdam',
    'tcga_gbm',
    'whas500',
    'pbc',
    'ovarian',
    'vlbw',
    'retinopathy'
]

def get_ds_name(ds: Dataset):
    match ds:
        case 'flchain':
            return 'FLC'
        case 'seer':
            return 'SEER'
        case 'support':
            return 'SUPPORT'
        case 'rotterdam':
            return 'Rotterdam'
        case 'tcga_gbm':
            return 'TCGA-GBM'
        case 'whas500':
            return 'WHAS500'
        case 'pbc':
            return 'PBC'
        case 'ovarian':
            return 'Ovarian'
        case 'vlbw':
            return 'VLBW'
        case 'retinopathy':
            return 'Retinopathy'
        case _:
            raise NotImplementedError(f"Unknown dataset {ds}")

def get_model_name(model: Model, f_latex: bool=False) -> str:
    match model:
        case "sdpm_mlp":
            return "SDPM"
        case "rsf":
            return "RSF"
        case "deepsurv":
            return "DeepSurv"
        case "deephit":
            return "DeepHit"
        case 'gbm_km': 
            return "GBM-KM"
        case 'gbm_wb': 
            return "GBM-Weibull"
        case _:
            raise NotImplementedError(f"Unable to find name for model {model}")
        
def get_metric_name(metric: Metric) -> str:
    match metric:
        case "auc":
            return "AUC"
        case "c_index":
            return "C-index"
        case "ibs":
            return "IBS"
        case _:
            raise NotImplementedError(f"Unable to find name for metric {metric}")

def get_metrics_list() -> List[Metric]:
    return list(get_args(Metric))

def get_models_list() -> List[Model]:
    return list(get_args(Model))
