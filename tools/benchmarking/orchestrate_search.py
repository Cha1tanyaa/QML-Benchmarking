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
import argparse
import inspect
import logging
import subprocess
from pathlib import Path

path_to_add_to_sys = Path(__file__).resolve().parents[2]
src_path = path_to_add_to_sys / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

REGRESSION_MODELS = {"LSTM", "QLSTM"}
DEFAULT_PHASE1_MODELS = {"Feedforward", "SVM", "XGBoost"}
DEFAULT_PHASE2_MODELS = {"LSTM", "QLSTM"}
DEFAULT_PHASE2_DATASETS = {"stock_tickerAAPL_train.csv"}
DEFAULT_PHASE3_DATASETS = {"credit_card_fraud_train.csv"}

#----------- Set up logging -----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#---------------------------------------

#------------------- Helper Functions -------------------


def _split_csv_values(value: str | None) -> set[str] | None:
    """Parse comma-separated CLI values into a normalized set."""
    if value is None:
        return None

    values = {item.strip() for item in value.split(",") if item.strip()}
    return values or None


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Orchestrate extension hyperparameter search phases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase1-models",
        type=str,
        default=",".join(sorted(DEFAULT_PHASE1_MODELS)),
        help="Comma-separated model names for phase 1.",
    )
    parser.add_argument(
        "--phase2-models",
        type=str,
        default=",".join(sorted(DEFAULT_PHASE2_MODELS)),
        help="Comma-separated model names for phase 2.",
    )
    parser.add_argument(
        "--phase2-datasets",
        type=str,
        default=",".join(sorted(DEFAULT_PHASE2_DATASETS)),
        help="Comma-separated phase-2 dataset file names (train CSV names).",
    )
    parser.add_argument(
        "--phase3-datasets",
        type=str,
        default=",".join(sorted(DEFAULT_PHASE3_DATASETS)),
        help="Comma-separated phase-3 dataset file names (train CSV names).",
    )
    return parser


def _run_phase(
    *,
    phase_name: str,
    models_to_run: list[str],
    dataset_paths: list[Path],
    runner_script: Path,
    output_root: Path,
    processed_combinations: set[tuple[str, str]],
) -> None:
    """Run one search phase and update processed combinations in place."""
    logging.info("\n--- %s: models=%s datasets=%s ---", phase_name, len(models_to_run), len(dataset_paths))
    if not models_to_run or not dataset_paths:
        logging.warning("%s has nothing to run.", phase_name)
        return

    for dataset_path in dataset_paths:
        dataset_name = dataset_path.name
        logging.info("%s - Dataset: %s", phase_name, dataset_name)
        for clf_name in models_to_run:
            if (dataset_name, clf_name) in processed_combinations:
                logging.warning("Skipping %s on %s (already processed).", clf_name, dataset_name)
                continue
            logging.info("%s search: %s on %s", phase_name, clf_name, dataset_name)
            if run_single_search(runner_script, clf_name, dataset_path, output_root):
                processed_combinations.add((dataset_name, clf_name))

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

def main(argv: list[str] | None = None):

    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    qml_benchmarks_root = path_to_add_to_sys
    import qml_benchmarks.models as models_module
    from qml_benchmarks.hyperparameter_settings import hyper_parameter_settings

    runner_script = qml_benchmarks_root / "tools" / "hyperparameter_search" / "grid_search.py"
    datasets_dir = qml_benchmarks_root / "data" / "generated" / "extension_datasets"

    if not runner_script.exists():
        raise FileNotFoundError(f"Runner script not found: {runner_script}")
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {datasets_dir}")

    hyperparam_results_root = qml_benchmarks_root / "results" / "extension" / "phase1"
    hyperparam_results_root.mkdir(parents=True, exist_ok=True)

    hyperparam_results2_root = qml_benchmarks_root / "results" / "extension" / "phase2"
    hyperparam_results2_root.mkdir(parents=True, exist_ok=True)

    hyperparam_results3_root = qml_benchmarks_root / "results" / "extension" / "phase3"
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

    all_dataset_files = sorted(datasets_dir.rglob("*_train.csv"))

    #------------------------------------------------------------

    #---------- Custom Settings for Specific Models and Datasets ----------
    phase1_models_config = _split_csv_values(args.phase1_models) or set(DEFAULT_PHASE1_MODELS)
    phase2_models_config = _split_csv_values(args.phase2_models) or set(DEFAULT_PHASE2_MODELS)
    phase2_dataset_names = _split_csv_values(args.phase2_datasets) or set(DEFAULT_PHASE2_DATASETS)
    phase3_dataset_names = _split_csv_values(args.phase3_datasets) or set(DEFAULT_PHASE3_DATASETS)

    phase1_models_to_run = sorted([m for m in phase1_models_config if m in all_model_names_with_settings])
    phase1_dataset_paths = sorted([p for p in all_dataset_files if p.name not in phase2_dataset_names])

    phase2_models_to_run = sorted([m for m in phase2_models_config if m in all_model_names_with_settings])
    phase2_dataset_paths = sorted(p for p in all_dataset_files if p.name in phase2_dataset_names)

    phase3_models_to_run = sorted([m for m in all_model_names_with_settings if m not in phase1_models_to_run and m not in phase2_models_to_run])
    phase3_dataset_paths = sorted([p for p in all_dataset_files if p.name in phase3_dataset_names])
    #----------------------------------------------------------------------

    processed_combinations: set[tuple[str, str]] = set()

    _run_phase(
        phase_name="PHASE 1",
        models_to_run=phase1_models_to_run,
        dataset_paths=phase1_dataset_paths,
        runner_script=runner_script,
        output_root=hyperparam_results_root,
        processed_combinations=processed_combinations,
    )
    _run_phase(
        phase_name="PHASE 2",
        models_to_run=phase2_models_to_run,
        dataset_paths=phase2_dataset_paths,
        runner_script=runner_script,
        output_root=hyperparam_results2_root,
        processed_combinations=processed_combinations,
    )
    _run_phase(
        phase_name="PHASE 3",
        models_to_run=phase3_models_to_run,
        dataset_paths=phase3_dataset_paths,
        runner_script=runner_script,
        output_root=hyperparam_results3_root,
        processed_combinations=processed_combinations,
    )

if __name__ == "__main__":
    main()
