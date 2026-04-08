# Copyright 2025 Chaitanya Agrawal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import logging
import time
from collections import defaultdict
import pandas as pd
from pathlib import Path

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

path_to_add_to_sys = Path(__file__).resolve().parents[2]
if str(path_to_add_to_sys) not in sys.path:
    sys.path.insert(0, str(path_to_add_to_sys))

import src.qml_benchmarks.models as models
from src.qml_benchmarks.hyperparam_search_utils import csv_to_dict, parse_hyperparameters, read_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_current_results(results_list: list[dict], root_path: Path, output_filename: str = "benchmark_best_hyperparams.csv") -> None:
    """Save collected benchmark results to a CSV file."""
    df_out = pd.DataFrame(results_list)
    output_dir = root_path / "paper_extension" / "results_phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_fp = output_dir / output_filename
    df_out.to_csv(out_fp, index=False)


def build_dataset_index(data_dir: Path) -> dict[str, dict[str, Path]]:
    """Index train/test CSV files by dataset stem to avoid repeated recursive scans."""
    dataset_index: dict[str, dict[str, Path]] = defaultdict(dict)
    for csv_file in data_dir.rglob("*.csv"):
        name = csv_file.name
        if name.endswith("_train.csv"):
            dataset_index[name[: -len("_train.csv")]]["train"] = csv_file
        elif name.endswith("_test.csv"):
            dataset_index[name[: -len("_test.csv")]]["test"] = csv_file
    return dataset_index


def parse_hp_filename(hp_file: Path) -> tuple[str, str]:
    """Extract model name and dataset stem from hyperparameter file naming convention."""
    stem = hp_file.stem.replace("-best-hyperparameters", "")
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Unexpected hyperparameter filename format: {hp_file.name}")
    model_name = parts[0]
    dataset_stem = "_".join(parts[1:-2])
    return model_name, dataset_stem


def compute_regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Compute regression metrics in a single helper for consistency."""
    return {
        "TestR2": r2_score(y_true, y_pred),
        "TestMSE": mean_squared_error(y_true, y_pred),
        "TestMAE": mean_absolute_error(y_true, y_pred),
    }

if __name__ == "__main__":

    qml_benchmarks_root = path_to_add_to_sys

    hyperparameter_dir   = qml_benchmarks_root / "paper_extension" / "results_phase2" / "results"
    data_dir = qml_benchmarks_root / "paper_extension" / "datasets_generated"
    all_results = []

    logging.info(f"Looking for hyperparam files in: {hyperparameter_dir}")
    hp_files = list(hyperparameter_dir.glob("*-best-hyperparameters.csv"))
    logging.info(f"Found {len(hp_files)} hyperparameter files.")

    dataset_index = build_dataset_index(data_dir)

    for hp_file in hp_files:
        model_name, dataset_stem = parse_hp_filename(hp_file)
        logging.info(f"Scoring {model_name} on {dataset_stem} (file: {hp_file.name})")

        dataset_files = dataset_index.get(dataset_stem, {})
        train_csv = dataset_files.get("train")
        test_csv = dataset_files.get("test")
        if train_csv is None or test_csv is None:
            logging.error("Missing train/test CSV for dataset stem '%s'.", dataset_stem)
            continue

        X_train, y_train = read_data(str(train_csv))
        X_test,  y_test  = read_data(str(test_csv))

        y_train = y_train.astype(float)
        y_test = y_test.astype(float)

        best_params = parse_hyperparameters(csv_to_dict(str(hp_file)))

        for seed in range(5):
            logging.info(f"Running seed {seed+1}/5 for {model_name} on {dataset_stem}")
            try:
                Model = getattr(models, model_name)
                clf = Model(**best_params, random_state=seed)
                start_train_time = time.perf_counter()
                clf.fit(X_train, y_train)
                training_time = time.perf_counter() - start_train_time
                train_r2 = clf.score(X_train, y_train)
                start_inference_time = time.perf_counter()
                y_pred_test = clf.predict(X_test)
                inference_time = time.perf_counter() - start_inference_time
                metrics = compute_regression_metrics(y_test, y_pred_test)
                all_results.append({
                    "Model": model_name,
                    "Dataset": dataset_stem,
                    "Seed": seed,
                    "TrainR2": train_r2,
                    **metrics,
                    "TrainingTime": training_time,
                    "InferenceTime": inference_time,
                })
            except (AttributeError, ValueError, TypeError, RuntimeError) as err:
                logging.error("Failed %s/%s seed %s: %s", model_name, dataset_stem, seed, err)
        save_current_results(all_results, qml_benchmarks_root) 