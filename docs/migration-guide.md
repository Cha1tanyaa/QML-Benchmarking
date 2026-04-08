# Migration Guide

## Major path changes

- `scripts/run_hyperparameter_search.py` -> `tools/hyperparameter_search/grid_search.py`
- `scripts/score_with_best_hyperparameters.py` -> `tools/hyperparameter_search/score_best.py`
- `paper_extension/benchmarks/extension_hyperparameter_search.py` -> `tools/benchmarking/orchestrate_search.py`
- `paper_extension/benchmarks/benchmark_pipeline_classifier.py` -> `tools/benchmarking/run_classification.py`
- `paper_extension/benchmarks/benchmark_pipeline_regressor.py` -> `tools/benchmarking/run_regression.py`
- `paper_extension/benchmarks/generate_datasets.py` -> `tools/data_generation/generate_extension_datasets.py`
- `paper_extension/benchmarks/automatic_hyperparmeter_search.py` -> `tools/legacy/automatic_hyperparameter_search.py`

## Data and result moves

- `paper_extension/datasets_generated/` -> `data/generated/extension_datasets/`
- `paper_extension/results_phase1/` -> `results/extension/phase1/`
- `paper_extension/results_phase2/` -> `results/extension/phase2/`
- `paper_extension/results_phase3/` -> `results/extension/phase3/`
- `paper/results/` -> `results/paper/original_study/`

## Utility module rename

- `qml_benchmarks.hyperparam_search_utils` -> `qml_benchmarks.hyperparameter_search_utils`
- A compatibility shim remains in `src/qml_benchmarks/hyperparam_search_utils.py`.
