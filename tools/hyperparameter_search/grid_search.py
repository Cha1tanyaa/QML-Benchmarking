# Copyright 2024 Xanadu Quantum Technologies Inc.
# Copyright 2025 Chaitanya Agrawal

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Chaitanya Agrawal from the original version.

"""Run hyperparameter search and store results with a command-line script."""

from __future__ import annotations

import argparse
import ast
import logging
import os
import random
import sys
import time
from importlib import import_module
from pathlib import Path
from typing import Any

root = Path(__file__).resolve().parents[2]
src = root / "src"
sys.path.insert(0, str(src))

random.seed(42)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def str_to_bool(value: str | bool) -> bool:
    """Parse bool-like CLI values while remaining compatible with '--flag True' usage."""
    if isinstance(value, bool):
        return value

    lowered = value.strip().lower()
    if lowered in {"1", "true", "t", "yes", "y"}:
        return True
    if lowered in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def _build_results_stem(classifier_name: str, dataset_path: Path) -> str:
    return f"{classifier_name}_{dataset_path.stem}_GridSearchCV"


def _parse_cli_value(value: str) -> Any:
    """Parse model-specific CLI overrides as literals when possible."""
    lowered = value.strip().lower()
    if lowered in {"true", "false", "none"}:
        return {"true": True, "false": False, "none": None}[lowered]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _log_run_settings(args: argparse.Namespace, hyperparam_grid: dict[str, Any]) -> None:
    logging.info("Running hyperparameter search with classifier=%s dataset=%s", args.classifier_name, args.dataset_path)
    logging.info("Scoring metrics=%s refit=%s", args.hyperparameter_scoring, args.hyperparameter_refit)
    logging.info("Hyperparam grid=%s", hyperparam_grid)


def _load_classifier(classifier_name: str):
    try:
        return getattr(import_module("qml_benchmarks.models"), classifier_name)
    except AttributeError as err:
        raise ValueError(f"Unknown classifier '{classifier_name}' in qml_benchmarks.models") from err


def main() -> None:
    logging.info("cpu count: %s", os.cpu_count())

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run experiments with hyperparameter search.",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--classifier-name",
        help="Classifier to run",
    )

    parser.add_argument(
        "--dataset-path",
        help="Path to the dataset",
    )

    parser.add_argument(
        "--results-path", default=".", help="Path to store the experiment results"
    )

    parser.add_argument(
        "--clean",
        help="True or False. Remove previous results if it exists",
        dest="clean",
        default=False,
        type=str_to_bool,
        nargs="?",
        const=True,
    )

    parser.add_argument(
        "--hyperparameter-scoring",
        # type=list,
        nargs="+",
        default=["accuracy", "roc_auc"],
        help="Scoring for hyperparameter search.",
    )

    parser.add_argument(
        "--hyperparameter-refit",
        type=str,
        default="accuracy",
        help="Refit scoring for hyperparameter search.",
    )

    parser.add_argument(
        "--plot-loss",
        help="True or False. Plot loss history for single fit",
        dest="plot_loss",
        default=False,
        type=str_to_bool,
        nargs="?",
        const=True,
    )

    parser.add_argument(
        "--n-jobs", type=int, default=-1, help="Number of parallel threads to run"
    )

    # Parse the arguments along with any extra arguments that might be model specific
    args, unknown_args = parser.parse_known_args()

    if any(arg is None for arg in [args.classifier_name,
                                   args.dataset_path]):
        msg = "\n================================================================================"
        msg += "\nA classifier from qml.benchmarks.model and dataset path are required. E.g., \n \n"
        msg += "python tools/hyperparameter_search/grid_search.py\n"
        msg += "  --classifier-name DataReuploadingClassifier\n"
        msg += "  --dataset-path train.csv\n"
        msg += "\nCheck all arguments for the script with \n"
        msg += "python tools/hyperparameter_search/grid_search.py --help\n"
        msg += "================================================================================"
        raise ValueError(msg)

    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    from qml_benchmarks.hyperparameter_search_utils import construct_hyperparameter_grid, read_data
    from qml_benchmarks.hyperparameter_settings import hyper_parameter_settings
    
    # Add model specific arguments to override the default hyperparameter grid
    hyperparam_grid = construct_hyperparameter_grid(
        hyper_parameter_settings, args.classifier_name
    )
    for hyperparam in hyperparam_grid:
        parser.add_argument(
            f"--{hyperparam}",
            type=str,
            nargs="+",
            default=None,
            help=f"{hyperparam} grid values for {args.classifier_name}",
        )

    args = parser.parse_args(unknown_args, namespace=args)

    for hyperparam in hyperparam_grid:
        override_values = getattr(args, hyperparam)
        if override_values is not None:
            hyperparam_grid[hyperparam] = [_parse_cli_value(item) for item in override_values]
    _log_run_settings(args, hyperparam_grid)

    experiment_path = Path(args.results_path)
    results_path = experiment_path / "results"

    results_path.mkdir(parents=True, exist_ok=True)

    ###################################################################
    # Get the classifier, dataset and search methods from the arguments
    ###################################################################
    Classifier = _load_classifier(args.classifier_name)
    classifier_name = Classifier.__name__

    # Run the experiments save the results
    dataset_path = Path(args.dataset_path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset path does not exist or is not a file: {dataset_path}")
    X, y = read_data(dataset_path)

    results_filename_stem = _build_results_stem(Classifier.__name__, dataset_path)
    result_file = results_path / f"{results_filename_stem}.csv"

    # If we have already run this experiment then continue
    if result_file.is_file():
        if args.clean is False:
            msg = "\n================================================================================="
            msg += f"\nResults exist in {result_file}"
            msg += "\nSpecify --clean True to override results or new --results-path"
            msg += "\n================================================================================="
            logging.warning(msg)
            print(msg)
            sys.exit(1)
        else:
            logging.warning("Cleaning existing results for %s", result_file)


    ###########################################################################
    # Single fit to check everything works
    ###########################################################################
    classifier = Classifier()
    start = time.perf_counter()
    classifier.fit(X, y)
    end = time.perf_counter()
    acc_train = classifier.score(X, y)
    logging.info(
        "%s Dataset path %s Train acc: %.6f Time single run %.6f",
        classifier_name,
        args.dataset_path,
        acc_train,
        end - start,
    )
    if hasattr(classifier, "loss_history_"):
        if args.plot_loss:
            import matplotlib.pyplot as plt

            plt.plot(classifier.loss_history_)
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.show()

    if hasattr(classifier, "n_qubits_"):
        logging.info(" ".join(["Num qubits", f"{classifier.n_qubits_}"]))

    ###########################################################################
    # Hyperparameter search
    ###########################################################################
    gs = GridSearchCV(estimator=classifier, param_grid=hyperparam_grid,
                        scoring=args.hyperparameter_scoring,
                        refit=args.hyperparameter_refit,
                        verbose=3,
                        n_jobs=args.n_jobs).fit(
        X, y
    )
    logging.info("Best hyperparams")
    logging.info(gs.best_params_)

    df = pd.DataFrame.from_dict(gs.cv_results_)
    df.to_csv(result_file)

    best_df = pd.DataFrame(list(gs.best_params_.items()), columns=['hyperparameter', 'best_value'])

    # Save best hyperparameters to a CSV file
    best_df.to_csv(results_path / f"{results_filename_stem}-best-hyperparameters.csv", index=False)


if __name__ == "__main__":
    main()
