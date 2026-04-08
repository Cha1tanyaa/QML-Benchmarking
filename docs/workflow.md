# Workflow

## 1) Generate extension datasets

```bash
python tools/data_generation/generate_extension_datasets.py
```

## 2) Run phased hyperparameter search

```bash
python tools/benchmarking/orchestrate_search.py
```

## 3) Run benchmark scoring

```bash
python tools/benchmarking/run_classification.py
python tools/benchmarking/run_regression.py
```

## 4) Optional single-run search/scoring

```bash
python tools/hyperparameter_search/grid_search.py --help
python tools/hyperparameter_search/score_best.py --help
```
