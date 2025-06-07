import sys
import logging
import pandas as pd
from pathlib import Path

path_to_add_to_sys = Path(__file__).resolve().parents[2]
if str(path_to_add_to_sys) not in sys.path:
    sys.path.insert(0, str(path_to_add_to_sys))

import src.qml_benchmarks.models as models
from src.qml_benchmarks.hyperparam_search_utils import read_data, csv_to_dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

if __name__ == "__main__":

    qml_benchmarks_root = path_to_add_to_sys

    hyperparameter_dir   = qml_benchmarks_root / "paper_extension" / "results_phase1" / "results"
    data_dir = qml_benchmarks_root / "paper_extension" / "datasets_generated"
    all_results = []

    logging.info(f"Looking for hyperparam files in: {hyperparameter_dir}")
    hp_files = list(hyperparameter_dir.glob("*-best-hyperparameters.csv"))
    logging.info(f"Found {len(hp_files)} hyperparameter files.")

    for hp_file in hp_files:
        stem = hp_file.stem.replace("-best-hyperparameters", "")
        parts = stem.split("_")

        classifier_name = parts[0]
        dataset_stem = "_".join(parts[1:-2])
        logging.info(f"Scoring {classifier_name} on {dataset_stem} (file: {hp_file.name})")

        train_csv = next(data_dir.rglob(f"{dataset_stem}_train.csv"))
        test_csv  = next(data_dir.rglob(f"{dataset_stem}_test.csv"))

        X_train, y_train = read_data(str(train_csv))
        X_test,  y_test  = read_data(str(test_csv))
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
            try:
                Model = getattr(models, classifier_name)
                clf   = Model(**best_params, random_state=seed)
                clf.fit(X_train, y_train)
                train_acc = clf.score(X_train, y_train)
                test_acc  = clf.score(X_test,  y_test)
                all_results.append({
                    "Model":         classifier_name,
                    "Dataset":       dataset_stem,
                    "Seed":          seed,
                    "TrainAccuracy": train_acc,
                    "TestAccuracy":  test_acc,
                })
            except Exception as e:
                logging.error(f"Failed {classifier_name}/{dataset_stem} seed {seed}: {e}")

    if all_results:
        df_out = pd.DataFrame(all_results)
        out_fp = qml_benchmarks_root / "paper_extension" / "results_phase1" / "benchmark_best_hyperparams.csv"
        df_out.to_csv(out_fp, index=False)
        logging.info(f"Saved benchmark results to {out_fp}")
    else:
        logging.warning("No benchmarks were run.")