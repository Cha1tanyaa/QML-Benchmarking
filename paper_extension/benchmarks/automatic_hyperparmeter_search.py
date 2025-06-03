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
def filter_compatible_models(dataset_stem, all_model_names, image_models, general_purpose_models, sequence_models, synthetic_dataset_stems):
    """
    Filters models based on dataset compatibility and hyperparameter settings.

    Args:
        dataset_stem (str): The stem of the dataset file name.
        all_model_names (list): List of all model names with defined hyperparameters.
        image_models (list): List of models suitable for image data.
        general_purpose_models (list): List of models suitable for general/tabular data.
        sequence_models (list): List of models suitable for sequence/time-series data.
        synthetic_dataset_stems (set): Set of synthetic dataset name patterns.

    Returns:
        list: Compatible models for the given dataset.
    """
    if "bars_and_stripes" in dataset_stem.lower():
        return [m for m in all_model_names if m in image_models or m in general_purpose_models]
    elif any(synth_name in dataset_stem.lower() for synth_name in synthetic_dataset_stems):
        return [m for m in all_model_names if m in general_purpose_models]
    elif "stock" in dataset_stem.lower():
        return [m for m in all_model_names if m in sequence_models or m in general_purpose_models]
    else:
        logging.info(f"Dataset {dataset_stem} did not match specific patterns, trying general purpose models with defined hyperparameters.")
        return [m for m in all_model_names if m in general_purpose_models]
#------------------- End of Helper Functions -------------------

def main():
    runner_script = repo / "scripts" / "run_hyperparameter_search.py"
    datasets_dir = repo / "results" / "datasets_generated"
    hyperparam_results_root = repo / "results" / "hyperparameter_search_results_auto"
    hyperparam_results_root.mkdir(parents=True, exist_ok=True)

    #---------- Discover Models and Datasets ----------
    all_model_names_from_module = [
        name for name, cls in inspect.getmembers(models_module, inspect.isclass)
        if cls.__module__.startswith("qml_benchmarks.models") and name != "BaseModel"
    ]
    logging.info(f"Found models in models module: {all_model_names_from_module}")

    all_model_names = [
        model_name for model_name in all_model_names_from_module
        if model_name in hyper_parameter_settings
    ]
    logging.info(f"Models with defined hyperparameters: {all_model_names}")

    skipped_models_no_settings = set(all_model_names_from_module) - set(all_model_names)
    if skipped_models_no_settings:
        logging.warning(f"Skipping models without defined hyperparameter settings: {skipped_models_no_settings}")
    #---------------------------------------------------

    #---------- Define Model Categories and Dataset Patterns ----------
    image_models = ["ConvolutionalNeuralNetwork", "WeiNet", "QuanvolutionalNeuralNetwork"]
    general_purpose_models = [
        "SVM", "MLP", "Perceptron", "XGBoost", "LogisticRegression",
        "CircuitCentricClassifier", "DataReuploadingClassifier",
        "DressedQuantumCircuitClassifier", "IQPVariationalClassifier",
        "QuantumMetricLearner", "QuantumBoltzmannMachine",
        "TreeTensorClassifier", "IQPKernelClassifier", "ProjectedQuantumKernel",
        "QuantumKitchenSinks", "SeparableVariationalClassifier", "SeparableKernelClassifier"
    ]
    sequence_models = ["LSTM", "QLSTM"]
    synthetic_dataset_stems = {"linearly_separable", "parity", "two_curves", "hidden_manifold"}

    dataset_files = list(datasets_dir.rglob("*.csv"))
    if not dataset_files:
        logging.warning(f"No CSV datasets found in {datasets_dir}")
        return

    logging.info(f"Found {len(dataset_files)} dataset(s) to process.")
    #------------------------------------------------------------

    #---------- Hyperparameter Search ----------
    for dataset_file_path in dataset_files:
        dataset_stem = dataset_file_path.stem
        logging.info(f"Processing dataset: {dataset_file_path.name}")

        compatible_models = filter_compatible_models(
            dataset_stem,
            all_model_names,
            image_models,
            general_purpose_models,
            sequence_models,
            synthetic_dataset_stems
        )

        if not compatible_models:
            logging.warning(f"No compatible models with defined hyperparameters identified for dataset: {dataset_stem}. Skipping.")
            continue

        logging.info(f"Compatible models for {dataset_stem}: {compatible_models}")

        for clf_name in compatible_models:
            logging.info(f"Preparing to run hyperparameter search for {clf_name} on {dataset_file_path.name}")

            cmd = [
                sys.executable,
                str(runner_script),
                "--classifier-name",    clf_name,
                "--dataset-path",       str(dataset_file_path),
                "--results-path",       str(hyperparam_results_root),
                "--n-jobs",             "-1",
                "--clean", "True"             
            ]

            logging.info(f"Executing command: {' '.join(cmd)}")

            try:
                process = subprocess.run(cmd, check=True, text=True, encoding='utf-8', capture_output=True)
                logging.info(f"Successfully ran hyperparameter search for {clf_name} on {dataset_file_path.name}.")
                if process.stdout:
                    logging.debug(f"Stdout for {clf_name} on {dataset_file_path.name}:\n{process.stdout.strip()}")
            except subprocess.CalledProcessError as e:
                logging.error(f"Error running hyperparameter search for {clf_name} on {dataset_file_path.name}.")
                logging.error(f"Command: {' '.join(e.cmd)}")
                logging.error(f"Return code: {e.returncode}")
                if e.stdout:
                    logging.error(f"Stdout: {e.stdout.strip()}")
                if e.stderr:
                    logging.error(f"Stderr: {e.stderr.strip()}")
            except FileNotFoundError:
                logging.error(f"Error: The script {runner_script} was not found. Ensure the path is correct.")
                return
            except Exception as e:
                logging.error(f"An unexpected error occurred while trying to run {clf_name} on {dataset_file_path.name}: {e}", exc_info=True)
    #------------------------------------------------

if __name__ == "__main__":
    main()