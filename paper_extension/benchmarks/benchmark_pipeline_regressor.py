import sys
import logging
import pandas as pd
from pathlib import Path

import time
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

path_to_add_to_sys = Path(__file__).resolve().parents[2]
if str(path_to_add_to_sys) not in sys.path:
    sys.path.insert(0, str(path_to_add_to_sys))

import src.qml_benchmarks.models as models
from src.qml_benchmarks.hyperparam_search_utils import read_data, csv_to_dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_current_results(results_list, root_path, output_filename="benchmark_best_hyperparams.csv"):
    """Saves the collected benchmark results to a CSV file, overwriting if it exists."""
    df_out = pd.DataFrame(results_list)
    output_dir = root_path / "paper_extension" / "results_phase2"
    output_dir.mkdir(parents=True, exist_ok=True) 
    out_fp = output_dir / output_filename
    df_out.to_csv(out_fp, index=False)

if __name__ == "__main__":

    qml_benchmarks_root = path_to_add_to_sys

    hyperparameter_dir   = qml_benchmarks_root / "paper_extension" / "results_phase2" / "results"
    data_dir = qml_benchmarks_root / "paper_extension" / "datasets_generated"
    all_results = []

    logging.info(f"Looking for hyperparam files in: {hyperparameter_dir}")
    hp_files = list(hyperparameter_dir.glob("*-best-hyperparameters.csv"))
    logging.info(f"Found {len(hp_files)} hyperparameter files.")

    total_hp_files = len(hp_files)

    for i, hp_file in enumerate(hp_files):
        stem = hp_file.stem.replace("-best-hyperparameters", "")
        parts = stem.split("_")

        model_name = parts[0]

        dataset_stem = "_".join(parts[1:-2])
        logging.info(f"Scoring {model_name} on {dataset_stem} (file: {hp_file.name})")

        train_csv = next(data_dir.rglob(f"{dataset_stem}_train.csv"))
        test_csv  = next(data_dir.rglob(f"{dataset_stem}_test.csv"))

        X_train, y_train = read_data(str(train_csv))
        X_test,  y_test  = read_data(str(test_csv))

        y_train = y_train.astype(float)
        y_test = y_test.astype(float)

        best_params = csv_to_dict(str(hp_file))

        processed_params = {}
        if best_params is not None:
            for key, value in best_params.items():
                if isinstance(value, str):
                    try:
                        float_val = float(value)
                        if float_val.is_integer():
                            processed_params[key] = int(float_val)
                        else:
                            processed_params[key] = float_val
                    except ValueError:
                        if value.lower() == 'true':
                            processed_params[key] = True
                        elif value.lower() == 'false':
                            processed_params[key] = False
                        elif value.lower() == 'none':
                            processed_params[key] = None
                        else:
                            processed_params[key] = value
                elif isinstance(value, float) and value.is_integer(): 
                    processed_params[key] = int(value)
                else:
                    processed_params[key] = value
        best_params = processed_params if best_params is not None else {}

        for seed in range(5):
            logging.info(f"Running seed {seed+1}/5 for {model_name} on {dataset_stem}")
            try:
                Model = getattr(models, model_name)
                clf   = Model(**best_params, random_state=seed)
                start_train_time = time.time()
                clf.fit(X_train, y_train)
                training_time = time.time() - start_train_time
                train_r2 = clf.score(X_train, y_train) 
                start_inference_time = time.time()
                y_pred_test = clf.predict(X_test)
                inference_time = time.time() - start_inference_time
                test_r2 = r2_score(y_test, y_pred_test)
                test_mse = mean_squared_error(y_test, y_pred_test)
                test_mae = mean_absolute_error(y_test, y_pred_test)
                all_results.append({
                    "Model": model_name,
                    "Dataset": dataset_stem,
                    "Seed": seed,
                    "TrainR2": train_r2,
                    "TestR2": test_r2,
                    "TestMSE": test_mse,
                    "TestMAE": test_mae,
                    "TrainingTime": training_time,
                    "InferenceTime": inference_time,
                })
            except Exception as e:
                logging.error(f"Failed {model_name}/{dataset_stem} seed {seed}: {e}")
        save_current_results(all_results, qml_benchmarks_root) 