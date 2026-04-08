# Architecture

## Top-level layout

- `src/qml_benchmarks/`: core Python package (models, data generators, shared utilities).
- `tools/`: executable workflows.
- `data/`: generated and raw datasets.
- `results/`: benchmark outputs.
- `paper/`: original-study assets (benchmarks/plots/scripts kept as legacy research material).
- `tests/`: repository-level tests.

## Tools layout

- `tools/hyperparameter_search/grid_search.py`: single dataset/model hyperparameter search.
- `tools/hyperparameter_search/score_best.py`: scoring with best hyperparameters.
- `tools/benchmarking/orchestrate_search.py`: phase orchestration for extension experiments.
- `tools/benchmarking/run_classification.py`: classification benchmark runner.
- `tools/benchmarking/run_regression.py`: regression benchmark runner.
- `tools/data_generation/generate_extension_datasets.py`: extension dataset generation.
- `tools/legacy/automatic_hyperparameter_search.py`: legacy broad auto-search runner.
