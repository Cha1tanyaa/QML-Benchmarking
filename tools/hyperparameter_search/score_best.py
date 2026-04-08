# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Score a model using the best hyperparameters, using a command-line script."""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from importlib import import_module
from pathlib import Path

root = Path(__file__).resolve().parents[1]
src = root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import pandas as pd
from qml_benchmarks.hyperparam_search_utils import csv_to_dict, parse_hyperparameters, read_data

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


def build_results_stem(classifier_name: str, trainset_path: Path) -> str:
    return f"{classifier_name}_{trainset_path.stem}_GridSearchCV"


def _require_args(args: argparse.Namespace) -> None:
    if any(arg is None for arg in [args.classifier_name, args.trainset_path, args.testset_path]):
        msg = "\n================================================================================"
        msg += "\nA classifier from qml.benchmarks.model and dataset path are required. E.g., \n \n"
        msg += "python score_with_best_hyperparameters.py\n"
        msg += "  --classifier-name DataReuploadingClassifier\n"
        msg += "  --trainset-path my_train_data.csv\n"
        msg += "  --testset-path my_test_data.csv\n"
        msg += "\nCheck all arguments for the script with \n"
        msg += "python score_with_best_hyperparameters.py --help\n"
        msg += "================================================================================"
        raise ValueError(msg)


def _load_classifier(classifier_name: str):
    try:
        return getattr(import_module("qml_benchmarks.models"), classifier_name)
    except AttributeError as err:
        raise ValueError(f"Unknown classifier '{classifier_name}' in qml_benchmarks.models") from err


if __name__ == "__main__":
    logging.info("cpu count: %s", os.cpu_count())

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Run experiments with hyperparameter search.",
                            formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--classifier-name",
        help="Classifier to run",
    )

    parser.add_argument(
        "--trainset-path",
        help="Path to the training set",
    )

    parser.add_argument(
        "--testset-path",
        help="Path to the test set",
    )

    parser.add_argument(
        "--hyperparams-path",
        default=".",
        help="Path to the file with the best hyperparameters",
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
        "--n-jobs", type=int, default=-1, help="Number of parallel threads to run"
    )

    # Parse the arguments along with any extra arguments that might be model specific
    args, unknown_args = parser.parse_known_args()

    _require_args(args)

    experiment_path = Path(args.results_path)
    results_path = experiment_path / "results"

    results_path.mkdir(parents=True, exist_ok=True)

    ###################################################################
    # Get the classifier, dataset and best hyperparameters
    ###################################################################
    Classifier = _load_classifier(args.classifier_name)
    classifier_name = Classifier.__name__

    # Load the data
    train_dataset_path = Path(args.trainset_path)
    test_dataset_path = Path(args.testset_path)
    hyperparams_path = Path(args.hyperparams_path)

    if not train_dataset_path.is_file():
        raise FileNotFoundError(f"Train dataset not found: {train_dataset_path}")
    if not test_dataset_path.is_file():
        raise FileNotFoundError(f"Test dataset not found: {test_dataset_path}")
    if not hyperparams_path.is_file():
        raise FileNotFoundError(f"Hyperparameter file not found: {hyperparams_path}")

    X_train, y_train = read_data(train_dataset_path)

    X_test, y_test = read_data(test_dataset_path)

    # Construct output path
    results_filename_stem = build_results_stem(Classifier.__name__, train_dataset_path)

    # If we have already run this experiment then continue
    path_out = results_path / f"{results_filename_stem}-best-hyperparams-results.csv"
    if path_out.is_file():
        if args.clean is False:
            msg = "\n================================================================================="
            msg += f"\nResults exist in {path_out}"
            msg += "\nSpecify --clean True to override results or new --results-path"
            msg += "\n================================================================================="
            logging.warning(msg)
            print(msg)
            sys.exit(1)
        else:
            logging.warning("Cleaning existing results for %s", path_out)

    # Load best hyperparameters
    best_hyperparams = parse_hyperparameters(csv_to_dict(hyperparams_path))

    # Score the model
    results_with_best_hyperparams = {"train_acc": [], "test_acc": []}
    for i in range(5):
        classifier = Classifier(**best_hyperparams, random_state=i)
        classifier.fit(X_train, y_train)

        acc_train = classifier.score(X_train, y_train)
        acc_test = classifier.score(X_test, y_test)
        results_with_best_hyperparams["train_acc"].append(acc_train)
        results_with_best_hyperparams["test_acc"].append(acc_test)

    logging.info("Results with best hyperparams %s", results_with_best_hyperparams)
    df = pd.DataFrame.from_dict(results_with_best_hyperparams)
    df.to_csv(path_out)
