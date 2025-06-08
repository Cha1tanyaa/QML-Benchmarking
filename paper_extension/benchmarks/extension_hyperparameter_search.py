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

    process = subprocess.run(cmd, check=True, text=True, encoding='utf-8', capture_output=True)
    logging.info(f"Successfully ran hyperparameter search for {clf_name} on {dataset_file_path.name}.")
#------------------- End of Helper Functions -------------------

def main():

    qml_benchmarks_root = path_to_add_to_sys

    runner_script = qml_benchmarks_root / "scripts" / "run_hyperparameter_search.py"
    datasets_dir = qml_benchmarks_root / "paper_extension" / "datasets_generated"

    hyperparam_results_root = qml_benchmarks_root / "paper_extension" / "results_phase1"
    hyperparam_results_root.mkdir(parents=True, exist_ok=True)

    hyperparam_results2_root = qml_benchmarks_root / "paper_extension" / "results_phase2"
    hyperparam_results2_root.mkdir(parents=True, exist_ok=True)

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

    #all_dataset_files = 
    #[p for p in all_dataset_files
    #if "bars_and_stripes" not in p.name.lower() and "hyperplanes" not in p.name.lower() and "hmm" not in p.name.lower() and "linsep" not in p.name.lower() and "two_curves" not #in p.name.lower()]
    #------------------------------------------------------------

    #---------- Custom Settings for Specific Models and Datasets ----------
    specific_models_for_all_datasets_raw = {"LSTM", "MLP", "QLSTM", "SVM", "XGBoost"}
    specific_models_for_all_datasets = [ m for m in specific_models_for_all_datasets_raw if m in all_model_names_with_settings]
    specific_datasets_for_all_models = {"stock_tickerAAPL_train.csv", "credit_card_card_fraud_train.csv"}
    #----------------------------------------------------------------------

    processed_combinations = set()
    #--------------------------------------------------------------------------

    #------------- Phase 1: Run specific models on ALL datasets ---------------
    logging.info(f"\n--- PHASE 1: Running models {specific_models_for_all_datasets} on ALL available datasets ---")
    for dataset_path in all_dataset_files:
        dataset_name = dataset_path.name
        logging.info(f"Phase 1 - Dataset: {dataset_name}")
        for clf_name in specific_models_for_all_datasets:
            logging.info(f"Phase 1 search: {clf_name} on {dataset_name}")
            run_single_search(runner_script, clf_name, dataset_path, hyperparam_results_root)
            processed_combinations.add((dataset_name, clf_name))
    #--------------------------------------------------------------------------

    #---------------- Phase 2: Run ALL models with settings on specific datasets ----------------
    logging.info(f"\n--- PHASE 2: Running ALL models with settings on specific datasets: {specific_datasets_for_all_models} ---")
    for target_dataset_basename in specific_datasets_for_all_models:
        target_dataset_path = None
        for d_path in all_dataset_files:
            if d_path.name == target_dataset_basename:
                target_dataset_path = d_path
                break
    
        logging.info(f"Phase 2 - Dataset: {target_dataset_basename}")
        logging.info(f"Phase 2 - Models: {all_model_names_with_settings}")

        for clf_name in all_model_names_with_settings:
            if (target_dataset_basename, clf_name) in processed_combinations:
                continue

            logging.info(f"Phase 2 search: {clf_name} on {target_dataset_basename}")
            run_single_search(runner_script, clf_name, target_dataset_path, hyperparam_results2_root)
            processed_combinations.add((target_dataset_basename, clf_name))
    #----------------------------------------------------------------------------

if __name__ == "__main__":
    main()