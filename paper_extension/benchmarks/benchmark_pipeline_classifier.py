import sys
import logging
import pandas as pd
from pathlib import Path

import ast
import time
import numpy as np
from sklearn.metrics import roc_auc_score, classification_report, accuracy_score

path_to_add_to_sys = Path(__file__).resolve().parents[2]
if str(path_to_add_to_sys) not in sys.path:
    sys.path.insert(0, str(path_to_add_to_sys))

import src.qml_benchmarks.models as models
from src.qml_benchmarks.hyperparam_search_utils import read_data, csv_to_dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_current_results(results_list, root_path, output_filename="benchmark_best_hyperparams.csv"):
    """Saves the collected benchmark results to a CSV file, overwriting if it exists."""
    df_out = pd.DataFrame(results_list)
    output_dir = root_path / "paper_extension" / "results_phase1"
    output_dir.mkdir(parents=True, exist_ok=True) 
    out_fp = output_dir / output_filename
    df_out.to_csv(out_fp, index=False)

if __name__ == "__main__":

    qml_benchmarks_root = path_to_add_to_sys

    hyperparameter_dir   = qml_benchmarks_root / "paper_extension" / "results_phase1" / "results"
    data_dir = qml_benchmarks_root / "paper_extension" / "datasets_generated"
    all_results = []

    logging.info(f"Looking for hyperparam files in: {hyperparameter_dir}")
    hp_files = list(hyperparameter_dir.glob("*-best-hyperparameters.csv"))
    logging.info(f"Found {len(hp_files)} hyperparameter files.")

    total_hp_files = len(hp_files)

    for i, hp_file in enumerate(hp_files):
        stem = hp_file.stem.replace("-best-hyperparameters", "")
        parts = stem.split("_")

        classifier_name = parts[0]


        dataset_stem = "_".join(parts[1:-2])
        logging.info(f"Scoring {classifier_name} on {dataset_stem} (file: {hp_file.name})")

        train_csv = next(data_dir.rglob(f"{dataset_stem}_train.csv"))
        test_csv  = next(data_dir.rglob(f"{dataset_stem}_test.csv"))

        X_train, y_train = read_data(str(train_csv))
        X_test,  y_test  = read_data(str(test_csv))

        y_train = y_train.astype(int)
        y_test = y_test.astype(int)

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
                            try:
                                processed_params[key] = ast.literal_eval(value)
                            except (ValueError, SyntaxError):
                                processed_params[key] = value
                elif isinstance(value, float) and value.is_integer(): 
                    processed_params[key] = int(value)
                else:
                    processed_params[key] = value
        best_params = processed_params if best_params is not None else {}

        for seed in range(5):
            logging.info(f"Running seed {seed+1}/5 for {classifier_name} on {dataset_stem}")
            try:
                Model = getattr(models, classifier_name)
                clf   = Model(**best_params, random_state=seed)
                start_train_time = time.time()
                clf.fit(X_train, y_train)
                training_time = time.time() - start_train_time
                train_acc = clf.score(X_train, y_train)
                start_inference_time = time.time()
                y_pred_test = clf.predict(X_test)
                inference_time = time.time() - start_inference_time
                test_acc = accuracy_score(y_test, y_pred_test)
                roc_auc = np.nan
                y_pred_proba_test = clf.predict_proba(X_test)
                if y_pred_proba_test.shape[1] == 2: 
                    roc_auc = roc_auc_score(y_test, y_pred_proba_test[:, 1])
                else:
                    roc_auc = roc_auc_score(y_test, y_pred_proba_test, multi_class='ovr', average='weighted')
                report_dict = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
                
                precision_weighted = report_dict['weighted avg']['precision']
                recall_weighted = report_dict['weighted avg']['recall']
                f1_weighted = report_dict['weighted avg']['f1-score']
                
                precision_macro = report_dict['macro avg']['precision']
                recall_macro = report_dict['macro avg']['recall']
                f1_macro = report_dict['macro avg']['f1-score']
                all_results.append({
                    "Model": classifier_name,
                    "Dataset": dataset_stem,
                    "Seed": seed,
                    "TrainAccuracy": train_acc,
                    "TestAccuracy": test_acc,
                    "TrainingTime": training_time,
                    "InferenceTime": inference_time,
                    "ROCAUC": roc_auc,
                    "Precision_weighted": precision_weighted,
                    "Recall_weighted": recall_weighted,
                    "F1_weighted": f1_weighted,
                    "Precision_macro": precision_macro,
                    "Recall_macro": recall_macro,
                    "F1_macro": f1_macro,
                })
            except Exception as e:
                logging.error(f"Failed {classifier_name}/{dataset_stem} seed {seed}: {e}")
        save_current_results(all_results, qml_benchmarks_root) 