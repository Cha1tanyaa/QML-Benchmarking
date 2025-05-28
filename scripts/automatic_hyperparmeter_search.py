import sys
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
src  = repo / "src"
sys.path.insert(0, str(src))

import subprocess
import inspect
import pandas as pd

import qml_benchmarks.models    as models_module

def main():
    repo          = Path(__file__).resolve().parents[1]
    runner        = repo / "scripts" / "run_hyperparameter_search.py"
    datasets_dir  = repo / "results" / "datasets_generated"
    hyperparam_results_root = repo / "results" / "hyperparameter_search_results"
    hyperparam_results_root.mkdir(parents=True, exist_ok=True)

    #  Find all classifier classes
    model_names = [
        name for name, cls in inspect.getmembers(models_module, inspect.isclass)
        if cls.__module__.startswith("qml_benchmarks.models")
    ]

    # Define model categories
    image_models = [
        "ConvolutionalNeuralNetwork", "WeiNet", "QuanvolutionalNeuralNetwork"
    ]
    flat_models = ["SVC", "MLPClassifier", "Perceptron", "XGBoostClassifier"]
    circuit_models = [
        "CircuitCentricClassifier", "DataReuploadingClassifier",
        "DressedQuantumCircuitClassifier", "IQPVariationalClassifier",
        "QuantumMetricLearner", "QuantumBoltzmannMachine",
        "TreeTensorClassifier"
    ]
    kernel_models = [
        "IQPKernelClassifier", "ProjectedQuantumKernel", "QuantumKitchenSinks"
    ]

    tabular_models = flat_models + ["LogisticRegression"]
    separable_models = [
        "SeparableVariationalClassifier", "SeparableKernelClassifier"
    ]

    sequence_models = ["LSTMClassifier", "QLSTMClassifier"]
    tree_models = ["XGBoostClassifier", "TreeTensorClassifier"]

    synthetic_list = {
        "linearly_separable", "parity", "two_curves", "hidden_manifold"
    }

    # Iterate over all datasets
    for dataset_file_path in datasets_dir.rglob("*.csv"):
        stem = dataset_file_path.stem
        print(f"Found dataset: {stem}")

        # Determine compatible models
        if "bars_and_stripes" in stem:
            allowed = image_models + flat_models + circuit_models + kernel_models
        elif stem in synthetic_list:
            allowed = tabular_models + circuit_models + kernel_models + separable_models
        elif "stock_features" in stem:
            allowed = sequence_models + tree_models
        else:
            print(f" → No compatible models for {stem}, skipping.")
            continue

        run_models = [m for m in model_names if m in allowed]
        print(f" → Will run models: {run_models}")

        for clf_name in run_models:
            cmd = [
                sys.executable,
                str(runner),
                "--classifier-name", clf_name,
                "--dataset-path",    str(dataset_file_path),
                "--results-path",    str(hyperparam_results_root),
                "--clean",           "True",
                "--n-jobs",          "-1",
            ]
            print(f"→ Running: {' '.join(cmd)}")
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
                print(f" Successfully ran hyperparameter search for {clf_name} on {dataset_file_path.name}")
            except subprocess.CalledProcessError as e:
                print(f" Error running hyperparameter search for {clf_name} on {dataset_file_path.name}:")

if __name__ == "__main__":
    main()