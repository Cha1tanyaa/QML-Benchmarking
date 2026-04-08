# QML Benchmarking

This repository benchmarks quantum and classical machine-learning models on synthetic and real-world datasets.

## Repository layout

- `src/qml_benchmarks/`: core package (models, data generation, shared settings/utilities).
- `tools/`: executable workflows for dataset generation, hyperparameter search, orchestration, and scoring.
- `data/`: raw and generated datasets.
- `results/`: benchmark outputs for extension phases and original study artifacts.
- `paper/`: legacy original-study scripts/assets.
- `docs/`: architecture, workflow, datasets, results, and migration notes.
- `tests/`: repository-level tests.

## Installation

```bash
pip install -e .
pip install -r requirements.txt
```

## Main commands

```bash
python tools/data_generation/generate_extension_datasets.py
python tools/benchmarking/orchestrate_search.py
python tools/benchmarking/run_classification.py
python tools/benchmarking/run_regression.py
```

Optional orchestration filters:

```bash
python tools/benchmarking/orchestrate_search.py \
	--phase1-models Feedforward,SVM,XGBoost \
	--phase2-models LSTM,QLSTM \
	--phase2-datasets stock_tickerAAPL_train.csv \
	--phase3-datasets credit_card_fraud_train.csv
```

Optional single-run tools:

```bash
python tools/hyperparameter_search/grid_search.py --help
python tools/hyperparameter_search/score_best.py --help
```

## Documentation

- `docs/architecture.md`
- `docs/workflow.md`
- `docs/datasets.md`
- `docs/results.md`
- `docs/migration-guide.md`
