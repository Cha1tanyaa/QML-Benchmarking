import sys
import inspect
import logging
import subprocess
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
src = repo / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

import qml_benchmarks.models as models_module
from qml_benchmarks.hyperparameter_settings import hyper_parameter_settings

#----------- Set up logging -----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#---------------------------------------

#------------------- Helper Functions -------------------
def filter_compatible_models(dataset_stem, all_model_names_with_settings, image_models, general_purpose_models, sequence_models, synthetic_dataset_stems):
    """
    Filters models based on dataset compatibility.
    Assumes all_model_names_with_settings already contains only models with defined hyperparameters.
    """
    compatible_subset = []
    if "bars_and_stripes" in dataset_stem.lower():
        compatible_subset = [m for m in all_model_names_with_settings if m in image_models or m in general_purpose_models]
    elif any(synth_name in dataset_stem.lower() for synth_name in synthetic_dataset_stems):
        compatible_subset = [m for m in all_model_names_with_settings if m in general_purpose_models]
    elif "stock" in dataset_stem.lower():
        compatible_subset = [m for m in all_model_names_with_settings if m in sequence_models or m in general_purpose_models]
    elif "creditcard" in dataset_stem.lower(): 
         compatible_subset = [m for m in all_model_names_with_settings if m in general_purpose_models]
    else:
        logging.info(f"Dataset {dataset_stem} did not match specific patterns, trying general purpose models with defined hyperparameters.")
        compatible_subset = [m for m in all_model_names_with_settings if m in general_purpose_models]
    
    return [m for m in compatible_subset if m in all_model_names_with_settings]

def run_single_search(runner_script_path, clf_name, dataset_file_path, hyperparam_results_root_path):
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

    logging.info(f"Executing command: {' '.join(cmd)}")
    try:
        process = subprocess.run(cmd, check=True, text=True, encoding='utf-8', capture_output=True)
        logging.info(f"Successfully ran hyperparameter search for {clf_name} on {dataset_file_path.name}.")
        if process.stdout:
            logging.debug(f"Stdout for {clf_name} on {dataset_file_path.name}:\n{process.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running hyperparameter search for {clf_name} on {dataset_file_path.name}.")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Return code: {e.returncode}")
        if e.stdout:
            logging.error(f"Stdout: {e.stdout.strip()}")
        if e.stderr:
            logging.error(f"Stderr: {e.stderr.strip()}")
        return False
    except FileNotFoundError:
        logging.error(f"Error: The script {runner_script_path} was not found. Ensure the path is correct.")
        return False
    except Exception as e:
        logging.error(f"An unexpected error occurred while trying to run {clf_name} on {dataset_file_path.name}: {e}", exc_info=True)
        return False
#------------------- End of Helper Functions -------------------

def main():
    runner_script = repo / "scripts" / "run_hyperparameter_search.py"
    datasets_dir = repo / "paper_extension" / "datasets_generated"
    hyperparam_results_root = repo / "paper_extension" / "results"
    hyperparam_results_root.mkdir(parents=True, exist_ok=True)

    #---------- Discover Models and Datasets ----------
    all_model_names_from_module = [
        name for name, cls in inspect.getmembers(models_module, inspect.isclass)
        if cls.__module__.startswith("qml_benchmarks.models") and name != "BaseModel"
    ]
    logging.info(f"Found models in models module: {all_model_names_from_module}")

    all_model_names_with_settings = [
        model_name for model_name in all_model_names_from_module
        if model_name in hyper_parameter_settings
    ]
    logging.info(f"Models with defined hyperparameters: {all_model_names_with_settings}")

    skipped_models_no_settings = set(all_model_names_from_module) - set(all_model_names_with_settings)
    if skipped_models_no_settings:
        logging.warning(f"Skipping models without defined hyperparameter settings: {skipped_models_no_settings}")

    all_dataset_files = list(datasets_dir.rglob("*.csv"))
    if not all_dataset_files:
        logging.warning(f"No CSV datasets found in {datasets_dir}")
        return
    logging.info(f"Found {len(all_dataset_files)} dataset(s) to process in total.")
    #------------------------------------------------------------

    #---------- Custom Settings for Specific Models and Datasets ----------
    specific_models_for_all_datasets_raw = {"LSTM", "MLP", "QLSTM", "SVM", "XGBoost"}
    specific_models_for_all_datasets = [
        m for m in specific_models_for_all_datasets_raw if m in all_model_names_with_settings
    ]
    skipped_specific = specific_models_for_all_datasets_raw - set(specific_models_for_all_datasets)
    if skipped_specific:
        logging.warning(f"Some specific models were requested but lack hyperparameter settings: {skipped_specific}")
    specific_datasets_for_all_models = {"stock_features.csv", "creditcard.csv"}
    #----------------------------------------------------------------------

    #------------------ Model Categories and Dataset Patterns ----------------
    image_models = ["ConvolutionalNeuralNetwork", "WeiNet", "QuanvolutionalNeuralNetwork"]
    general_purpose_models = [
        "SVM", "MLP", "Perceptron", "XGBoost", "LogisticRegression",
        "CircuitCentricClassifier", "DataReuploadingClassifier",
        "DressedQuantumCircuitClassifier", "IQPVariationalClassifier",
        "QuantumMetricLearner", "QuantumBoltzmannMachine", "SVC",
        "TreeTensorClassifier", "IQPKernelClassifier", "ProjectedQuantumKernel",
        "QuantumKitchenSinks", "SeparableVariationalClassifier", "SeparableKernelClassifier"
    ]
    sequence_models = ["LSTM", "QLSTM"]
    synthetic_dataset_stems = {"linearly_separable", "parity", "two_curves", "hidden_manifold"}

    processed_combinations = set()
    #--------------------------------------------------------------------------

    #------------- Phase 1: Run specific models on ALL datasets ---------------
    logging.info(f"\n--- PHASE 1: Running models {specific_models_for_all_datasets} on ALL available datasets ---")
    for dataset_path in all_dataset_files:
        dataset_name = dataset_path.name
        logging.info(f"Phase 1 - Processing dataset: {dataset_name}")
        for clf_name in specific_models_for_all_datasets:
            if (dataset_name, clf_name) in processed_combinations:
                logging.debug(f"Skipping already processed (Phase 1 target): {clf_name} on {dataset_name}")
                continue
            
            logging.info(f"Attempting Phase 1 search: {clf_name} on {dataset_name}")
            run_single_search(runner_script, clf_name, dataset_path, hyperparam_results_root)
            processed_combinations.add((dataset_name, clf_name))
    #--------------------------------------------------------------------------

    #---------------- Phase 2: Run ALL compatible models on specific datasets ----------------
    logging.info(f"\n--- PHASE 2: Running ALL compatible models on specific datasets: {specific_datasets_for_all_models} ---")
    for target_dataset_basename in specific_datasets_for_all_models:
        target_dataset_path = None
        for d_path in all_dataset_files:
            if d_path.name == target_dataset_basename:
                target_dataset_path = d_path
                break
        
        if not target_dataset_path:
            logging.warning(f"Phase 2 - Target dataset '{target_dataset_basename}' not found in {datasets_dir}. Skipping.")
            continue

        dataset_stem = target_dataset_path.stem
        logging.info(f"Phase 2 - Processing dataset: {target_dataset_basename}")

        compatible_models = filter_compatible_models(
            dataset_stem,
            all_model_names_with_settings,
            image_models,
            general_purpose_models,
            sequence_models,
            synthetic_dataset_stems
        )
        
        logging.info(f"Phase 2 - Compatible models for {target_dataset_basename}: {compatible_models}")

        for clf_name in compatible_models:
            if (target_dataset_basename, clf_name) in processed_combinations:
                logging.debug(f"Skipping already processed (Phase 2 target): {clf_name} on {target_dataset_basename}")
                continue

            logging.info(f"Attempting Phase 2 search: {clf_name} on {target_dataset_basename}")
            run_single_search(runner_script, clf_name, target_dataset_path, hyperparam_results_root)
            processed_combinations.add((target_dataset_basename, clf_name))
    #----------------------------------------------------------------------------

if __name__ == "__main__":
    main()