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
import inspect
import logging
import subprocess
from pathlib import Path

path_to_add_to_sys = Path(__file__).resolve().parents[2]
if str(path_to_add_to_sys) not in sys.path:
    sys.path.insert(0, str(path_to_add_to_sys))

import src.qml_benchmarks.models as models_module
from src.qml_benchmarks.hyperparameter_settings import hyper_parameter_settings

REGRESSION_MODELS = {"LSTM", "QLSTM"}

#----------- Set up logging -----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#---------------------------------------

#------------------- Helper Functions -------------------

def run_single_search(
    runner_script_path: Path,
    clf_name: str,
    dataset_file_path: Path,
    hyperparam_results_root_path: Path,
) -> bool:
    """
    Executes a single hyperparameter search subprocess.
    Returns True on success, False on failure.
    """
    cmd = [
        sys.executable,
        str(runner_script_path),
        "--classifier-name", clf_name,
        "--dataset-path", str(dataset_file_path),
        "--results-path", str(hyperparam_results_root_path),
        "--n-jobs", "-1",
        "--clean", "True"
    ]

    if clf_name in REGRESSION_MODELS:
        cmd.extend([
            "--hyperparameter-scoring", "r2", "neg_mean_squared_error",
            "--hyperparameter-refit", "r2"
        ])

    logging.info(f"Executing command: {' '.join(cmd)}")

    try:
        process = subprocess.run(cmd, check=True, text=True, encoding='utf-8', capture_output=True)
        logging.info(f"Successfully ran hyperparameter search for {clf_name} on {dataset_file_path.name}.")
        logging.debug(f"Stdout for {clf_name} on {dataset_file_path.name}:\n{process.stdout.strip()}")
        if process.stderr:
            logging.warning(f"Subprocess stderr for {clf_name} on {dataset_file_path.name}:\n{process.stderr.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running hyperparameter search for {clf_name} on {dataset_file_path.name}.")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Return code: {e.returncode}")
        if e.output:
            logging.error(f"Stdout: {e.output.strip()}")
        if e.stderr:
            logging.error(f"Stderr: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        logging.error(f"Error: The script {runner_script_path} was not found. Ensure the path is correct.")
        return False
    except (OSError, UnicodeError) as exc:
        logging.error(
            f"An unexpected error occurred while trying to run {clf_name} on {dataset_file_path.name}: {exc}",
            exc_info=True,
        )
        return False
#------------------- End of Helper Functions -------------------

def main():

    qml_benchmarks_root = path_to_add_to_sys

    runner_script = qml_benchmarks_root / "scripts" / "run_hyperparameter_search.py"
    datasets_dir = qml_benchmarks_root / "paper_extension" / "datasets_generated"

    if not runner_script.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_script}")
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {datasets_dir}")

    hyperparam_results_root = qml_benchmarks_root / "paper_extension" / "results_phase1"
    hyperparam_results_root.mkdir(parents=True, exist_ok=True)

    hyperparam_results2_root = qml_benchmarks_root / "paper_extension" / "results_phase2"
    hyperparam_results2_root.mkdir(parents=True, exist_ok=True)

    hyperparam_results3_root = qml_benchmarks_root / "paper_extension" / "results_phase3"
    hyperparam_results3_root.mkdir(parents=True, exist_ok=True)

    #---------- Discover Models and Datasets ----------
    all_model_names_from_module = [
        name for name, cls in inspect.getmembers(models_module, inspect.isclass)
        if cls.__module__.startswith("qml_benchmarks.models") and name != "BaseModel"
    ]
    logging.info(f"Found models: {all_model_names_from_module}")

    all_model_names_with_settings = [
        model_name for model_name in all_model_names_from_module
        if model_name in hyper_parameter_settings
    ]

    all_dataset_files = list(datasets_dir.rglob("*_train.csv"))

    #------------------------------------------------------------

    #---------- Custom Settings for Specific Models and Datasets ----------
    phase1_models_config = {"Feedforward", "SVM", "XGBoost"}
    phase2_models_config = {"LSTM", "QLSTM"}
    phase2_dataset_paths = {"stock_tickerAAPL_train.csv"}
    phase3_dataset_paths = {"credit_card_fraud_train.csv"} 

    phase1_models_to_run = sorted([m for m in phase1_models_config if m in all_model_names_with_settings])
    phase1_dataset_paths = sorted([p for p in all_dataset_files if p.name not in phase2_dataset_paths])

    phase2_models_to_run = sorted([m for m in phase2_models_config if m in all_model_names_with_settings])
    phase2_dataset_paths = sorted(p for p in all_dataset_files if p.name in phase2_dataset_paths) 

    phase3_models_to_run = sorted([m for m in all_model_names_with_settings if m not in phase1_models_to_run and m not in phase2_models_to_run])
    phase3_dataset_paths = sorted([p for p in all_dataset_files if p.name in phase3_dataset_paths])
    #----------------------------------------------------------------------

    processed_combinations: set[tuple[str, str]] = set()

    #------------- Phase 1: Run specific models on ALL datasets ---------------
    logging.info(f"\n--- PHASE 1: Running New models {phase1_models_to_run} ---")
    for dataset_path in phase1_dataset_paths:
        dataset_name = dataset_path.name
        logging.info(f"Phase 1 - Dataset: {dataset_name}")
        for clf_name in phase1_models_to_run:
            logging.info(f"Phase 1 search: {clf_name} on {dataset_name}")
            if run_single_search(runner_script, clf_name, dataset_path, hyperparam_results_root):
                processed_combinations.add((dataset_name, clf_name))
    #--------------------------------------------------------------------------

    #------------- Phase 2: Run specific models on ALL datasets ---------------
    logging.info(f"\n--- PHASE 2: Running Regression models {phase2_models_to_run} ---")
    for dataset_path in phase2_dataset_paths:
        dataset_name = dataset_path.name
        logging.info(f"Phase 2 - Dataset: {dataset_name}")
        for clf_name in phase2_models_to_run:
            logging.info(f"Phase 2 search: {clf_name} on {dataset_name}")
            if run_single_search(runner_script, clf_name, dataset_path, hyperparam_results2_root):
                processed_combinations.add((dataset_name, clf_name))
    #--------------------------------------------------------------------------

    #---------------- Phase 3: Run ALL models with settings on specific datasets ----------------
    logging.info(f"\n--- PHASE 3: Running Other models: {phase3_models_to_run} ---")
    for dataset_path in phase3_dataset_paths:
            dataset_name = dataset_path.name
            logging.info(f"Phase 3 - Current Dataset: {dataset_name}")
            for clf_name in phase3_models_to_run:
                if (dataset_name, clf_name) in processed_combinations:
                    logging.warning(f"Skipping {clf_name} on {dataset_name} (already processed).")
                    continue
                logging.info(f"Phase 3 search: {clf_name} on {dataset_name}")
                if run_single_search(runner_script, clf_name, dataset_path, hyperparam_results3_root):
                    processed_combinations.add((dataset_name, clf_name))
    #----------------------------------------------------------------------------

if __name__ == "__main__":
    main()