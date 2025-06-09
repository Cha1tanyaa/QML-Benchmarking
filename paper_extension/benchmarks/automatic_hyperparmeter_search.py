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
    if any(synth_name in dataset_stem.lower() for synth_name in synthetic_dataset_stems):
        return [m for m in all_model_names if m in image_models or m in general_purpose_models]
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
    logging.info(f"Found models: {all_model_names_from_module}")

    all_model_names = [
        model_name for model_name in all_model_names_from_module
        if model_name in hyper_parameter_settings
    ]
    #---------------------------------------------------

    #---------- Define Model Categories and Dataset Patterns ----------
    image_models = ["ConvolutionalNeuralNetwork", "WeiNet", "QuanvolutionalNeuralNetwork"]
    general_purpose_models = [
        "SVM", "MLP", "XGBoost",
        "CircuitCentricClassifier", "DataReuploadingClassifier",
        "DressedQuantumCircuitClassifier", "IQPVariationalClassifier",
        "QuantumMetricLearner", "QuantumBoltzmannMachine",
        "TreeTensorClassifier", "IQPKernelClassifier", "ProjectedQuantumKernel",
        "QuantumKitchenSinks", "SeparableVariationalClassifier", "SeparableKernelClassifier"
    ]
    sequence_models = ["LSTM", "QLSTM"]
    synthetic_dataset_stems = {"linearly_separable", "hmm", "two_curves", "hidden_manifold"}

    dataset_files = list(datasets_dir.rglob("*.csv"))
    #------------------------------------------------------------

    #---------- Hyperparameter Search ----------
    for dataset_file_path in dataset_files:
        dataset_stem = dataset_file_path.stem
        logging.info(f"Dataset: {dataset_file_path.name}")

        compatible_models = filter_compatible_models(
            dataset_stem,
            all_model_names,
            image_models,
            general_purpose_models,
            sequence_models,
            synthetic_dataset_stems
        )

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

            if clf_name in ["LSTM", "QLSTM"]:
                cmd.extend([
                    "--hyperparameter-scoring", "r2", "neg_mean_squared_error",
                    "--hyperparameter-refit", "r2"
                ])

            logging.info(f"Executing command: {' '.join(cmd)}")

            process = subprocess.run(cmd, check=True, text=True, encoding='utf-8', capture_output=True)
            logging.info(f"Successfully ran hyperparameter search for {clf_name} on {dataset_file_path.name}.")
            if process.stdout:
                logging.debug(f"Stdout for {clf_name} on {dataset_file_path.name}:\n{process.stdout.strip()}")
    #------------------------------------------------

if __name__ == "__main__":
    main()