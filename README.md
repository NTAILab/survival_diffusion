# SDPM: Survival Diffusion Probabilistic Model for Continuous-Time Survival Analysis

This repository contains the code and preprocessed data used for the experiments in the article:

**SDPM: Survival Diffusion Probabilistic Model for Continuous-Time Survival Analysis**

[Preprint paper](https://arxiv.org/abs/2605.22776)

SDPM is a diffusion-based model for continuous-time survival analysis. The repository includes the SDPM model implementation, baseline wrappers, experiment runners, preprocessed benchmark datasets, and notebooks for additional analyses and table generation.

## Repository Contents

- `sdpm/` - Python package with the SDPM implementation and experiment utilities.
- `sdpm/sdpm.py` - main SDPM model class.
- `sdpm/diffusion/` - diffusion process, scheduler, and time-conditioning components.
- `sdpm/baseline/` - neural-network baseline components used by SDPM.
- `sdpm/experiments/` - comparison experiment code and baseline model wrappers.
- `sdpm/experiments/results/final_results.csv` - aggregated final experiment results.
- `data/` - preprocessed survival datasets used in the experiments.
- `comparison_script.sh` - convenience script for running the main comparison experiments.
- `ablation_K.ipynb`, `ablation_r.ipynb`, `ablation_sf.ipynb`, `log_delta_ablation.ipynb` - notebooks with ablation and additional analysis experiments.
- `final_tables.ipynb` - notebook for aggregating results and preparing final tables.
- `environment.yml` - Conda environment specification.

## Installation

Create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate sdpm
```

Run commands from the repository root so that the local `sdpm` package and `data/` directory are available.

## Data

The repository includes preprocessed `.npz` versions of the ten datasets used in the experiments:

- `flchain`
- `ovarian`
- `pbc`
- `retinopathy`
- `rotterdam`
- `seer`
- `support`
- `tcga_gbm`
- `vlbw`
- `whas500`

Each dataset is loaded by name from `data/<dataset>.npz`.

## Running Comparison Experiments

The main comparison experiments can be run with:

```bash
bash comparison_script.sh <dataset> [device]
```

For example:

```bash
bash comparison_script.sh vlbw cuda:0
```

If no device is provided, the script uses `cuda:0` for SDPM. Classical and non-SDPM baselines are run on CPU.

The script runs the following models:

- `sdpm_mlp` - proposed SDPM model
- `rsf` - Random Survival Forest
- `deepsurv` - DeepSurv
- `deephit` - DeepHit
- `gbm_wb` - XGBSEStackedWeibull
- `gbm_km` - XGBSEKaplanNeighbors

Results are appended to:

```text
sdpm/experiments/results/results.csv
```

The comparison runner can also be invoked directly:

```bash
python -m sdpm.experiments.comparison \
  -data vlbw \
  -model sdpm_mlp \
  -device cuda:0 \
  -trials 100 \
  -repeats 10 \
  -threads 16 \
  -table_filename sdpm/experiments/results/results.csv
```

Available dataset names are listed above. Available model names are `sdpm_mlp`, `rsf`, `deepsurv`, `deephit`, `gbm_wb`, and `gbm_km`.

## Experimental Setup

The comparison experiments evaluate SDPM against five survival-analysis baselines on ten real-world datasets. The evaluation uses repeated 4-fold cross-validation with Optuna hyperparameter optimization inside each fold.

The reported metrics are:

- Harrell's C-index
- Integrated time-dependent AUC
- Integrated Brier score (IBS)

The main experiment results used for the article are stored in:

```text
sdpm/experiments/results/final_results.csv
```

## Ablation Experiments and Tables

Additional experimental analyses are provided as Jupyter notebooks:

- `ablation_K.ipynb` - influence of the number of generated samples K.
- `ablation_r.ipynb` - influence of the number of diffusion steps r.
- `ablation_sf.ipynb` - survival-function related ablation analysis.
- `log_delta_ablation.ipynb` - additional ablation for the event label and time representation.
- `final_tables.ipynb` - result aggregation and final table preparation.

## Citation

```bibtex
@unpublished{kirpichenko2026sdpm,
      title={SDPM: Survival Diffusion Probabilistic Model for Continuous-Time Survival Analysis}, 
      author={Stanislav R. Kirpichenko and Andrei V. Konstantinov and Lev V. Utkin},
      year={2026},
      eprint={2605.22776},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.22776}, 
}
```
