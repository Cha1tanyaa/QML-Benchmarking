#import os
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

#---------------- Set up the environment for parallel processing (optional) ---------
#BLAS_THREADS = 48

# set before any numpy/sklearn import (optional if you only do it in the subprocess env)
#os.environ["OMP_NUM_THREADS"]       = str(BLAS_THREADS)
#os.environ["MKL_NUM_THREADS"]       = str(BLAS_THREADS)
#os.environ["OPENBLAS_NUM_THREADS"]  = str(BLAS_THREADS)
#os.environ["NUMEXPR_NUM_THREADS"]   = str(BLAS_THREADS)
#--------------------------------------------------------------------------------------

#----------- Set up logging -----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#---------------------------------------

#------------------- Helper Functions -------------------

def run_single_search(runner_script_path, clf_name, dataset_file_path, hyperparam_results_root_path):
    """
    Executes a single hyperparameter search subprocess.
    Returns True on success, False on failure.
    """

    #env = os.environ.copy()
    #env["OMP_NUM_THREADS"]      = str(BLAS_THREADS)
    #env["MKL_NUM_THREADS"]      = str(BLAS_THREADS)
    #env["OPENBLAS_NUM_THREADS"] = str(BLAS_THREADS)
    #env["NUMEXPR_NUM_THREADS"]  = str(BLAS_THREADS)

    cmd = [
        sys.executable,
        str(runner_script_path),
        "--classifier-name", clf_name,
        "--dataset-path", str(dataset_file_path),
        "--results-path", str(hyperparam_results_root_path),
        "--n-jobs", "-1",
        #"--n-jobs", str(BLAS_THREADS),
        "--clean", "True"
    ]

    if clf_name in ["LSTM", "QLSTM"]:
        cmd.extend([
            "--hyperparameter-scoring", "r2", "neg_mean_squared_error",
            "--hyperparameter-refit", "r2"
        ])

    try:
        process = subprocess.run(cmd, check=True, text=True, encoding='utf-8', capture_output=True)
        logging.info(f"Successfully ran hyperparameter search for {clf_name} on {dataset_file_path.name}.")
        logging.debug(f"Subprocess stdout for {clf_name} on {dataset_file_path.name}:\n{process.stdout}")
        if process.stderr:
            logging.warning(f"Subprocess stderr for {clf_name} on {dataset_file_path.name}:\n{process.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Subprocess for {clf_name} on {dataset_file_path.name} failed with exit code {e.returncode}.")
        logging.error(f"Command was: {' '.join(e.cmd)}")
        if e.stdout:
            logging.error(f"Subprocess stdout:\n{e.stdout}")
        if e.stderr:
            logging.error(f"Subprocess stderr:\n{e.stderr}")
        return False
#------------------- End of Helper Functions -------------------

def main():

    qml_benchmarks_root = path_to_add_to_sys

    runner_script = qml_benchmarks_root / "scripts" / "run_hyperparameter_search.py"
    datasets_dir = qml_benchmarks_root / "paper_extension" / "datasets_generated"

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

    processed_combinations = set()

    #------------- Phase 1: Run specific models on ALL datasets ---------------
    logging.info(f"\n--- PHASE 1: Running New models {phase1_models_to_run} ---")
    for dataset_path in phase1_dataset_paths:
        dataset_name = dataset_path.name
        logging.info(f"Phase 1 - Dataset: {dataset_name}")
        for clf_name in phase1_models_to_run:
            logging.info(f"Phase 1 search: {clf_name} on {dataset_name}")
            run_single_search(runner_script, clf_name, dataset_path, hyperparam_results_root)
            processed_combinations.add((dataset_name, clf_name))
    #--------------------------------------------------------------------------

    #------------- Phase 2: Run specific models on ALL datasets ---------------
    logging.info(f"\n--- PHASE 2: Running Regression models {phase2_models_to_run} ---")
    for dataset_path in phase2_dataset_paths:
        dataset_name = dataset_path.name
        logging.info(f"Phase 2 - Dataset: {dataset_name}")
        for clf_name in phase2_models_to_run:
            logging.info(f"Phase 2 search: {clf_name} on {dataset_name}")
            run_single_search(runner_script, clf_name, dataset_path, hyperparam_results2_root)
            processed_combinations.add((dataset_name, clf_name))
    #--------------------------------------------------------------------------

    #---------------- Phase 3: Run ALL models with settings on specific datasets ----------------
    logging.info(f"\n--- PHASE 3: Running Other models: {phase3_models_to_run} ---")
    for dataset_path in phase3_dataset_paths:
            dataset_name = dataset_path.name
            logging.info(f"Phase 3 - Current Dataset: {dataset_name}")
            for clf_name in phase3_models_to_run:
                if (dataset_name, clf_name) in processed_combinations:
                    logging.warning(f"Skipping {clf_name} on {dataset_name} (already processed.")
                    continue
                logging.info(f"Phase 3 search: {clf_name} on {dataset_name}")
                run_single_search(runner_script, clf_name, dataset_path, hyperparam_results3_root)
                processed_combinations.add((dataset_name, clf_name))
    #----------------------------------------------------------------------------

if __name__ == "__main__":
    main()