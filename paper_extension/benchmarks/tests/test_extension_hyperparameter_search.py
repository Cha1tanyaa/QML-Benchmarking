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

import pytest
import sys
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, call

path_to_add_to_sys = Path(__file__).resolve().parents[3]
if str(path_to_add_to_sys) not in sys.path:
    sys.path.insert(0, str(path_to_add_to_sys))

from paper_extension.benchmarks.extension_hyperparameter_search import run_single_search

#-------------------------- Pytest Fixtures --------------------------
@pytest.fixture
def mock_paths():
    """Provides mock Path objects for runner_script, dataset_file, and results_root."""
    return {
        "runner": MagicMock(spec=Path, __str__=lambda x: "mock_runner.py"),
        "dataset": MagicMock(spec=Path, name="mock_dataset_train.csv", __str__=lambda x: "path/to/mock_dataset_train.csv"),
        "results": MagicMock(spec=Path, __str__=lambda x: "path/to/results")
    }

@pytest.fixture
def mock_logging(mocker):
    """Mocks logging functions."""
    return {
        "info": mocker.patch("paper_extension.benchmarks.extension_hyperparameter_search.logging.info"),
        "error": mocker.patch("paper_extension.benchmarks.extension_hyperparameter_search.logging.error"),
        "debug": mocker.patch("paper_extension.benchmarks.extension_hyperparameter_search.logging.debug"),
        "warning": mocker.patch("paper_extension.benchmarks.extension_hyperparameter_search.logging.warning"),
    }
#--------------------------------------------------------------------

#-------------------------- Test: Successful Execution --------------------------
def test_run_single_search_success(mocker, mock_paths, mock_logging):
    """Tests successful execution of run_single_search."""
    mock_subprocess_run = mocker.patch("subprocess.run")
    mock_process = MagicMock()
    mock_process.stdout = "Subprocess output"
    mock_subprocess_run.return_value = mock_process

    clf_name = "TestClassifier"
    
    result = run_single_search(
        mock_paths["runner"],
        clf_name,
        mock_paths["dataset"],
        mock_paths["results"]
    )

    assert result is True
    expected_cmd = [
        sys.executable,
        str(mock_paths["runner"]),
        "--classifier-name", clf_name,
        "--dataset-path", str(mock_paths["dataset"]),
        "--results-path", str(mock_paths["results"]),
        "--n-jobs", "-1",
        "--clean", "True"
    ]
    mock_subprocess_run.assert_called_once_with(
        expected_cmd, check=True, text=True, encoding='utf-8', capture_output=True
    )
    mock_logging["info"].assert_any_call(f"Executing command: {' '.join(expected_cmd)}")
    mock_logging["info"].assert_any_call(f"Successfully ran hyperparameter search for {clf_name} on {mock_paths['dataset'].name}.")
    mock_logging["debug"].assert_any_call(f"Stdout for {clf_name} on {mock_paths['dataset'].name}:\n{mock_process.stdout.strip()}")
#---------------------------------------------------------------------------------

#-------------------------- Test: CalledProcessError --------------------------
def test_run_single_search_called_process_error(mocker, mock_paths, mock_logging):
    """Tests run_single_search when subprocess.run raises CalledProcessError."""
    mock_subprocess_run = mocker.patch("subprocess.run")
    error_cmd = ["some", "command"]
    error_instance = subprocess.CalledProcessError(
        returncode=1, cmd=error_cmd, output="stdout error", stderr="stderr error"
    )
    mock_subprocess_run.side_effect = error_instance

    clf_name = "ErrorClassifier"

    result = run_single_search(
        mock_paths["runner"],
        clf_name,
        mock_paths["dataset"],
        mock_paths["results"]
    )

    assert result is False
    mock_logging["error"].assert_any_call(f"Error running hyperparameter search for {clf_name} on {mock_paths['dataset'].name}.")
    mock_logging["error"].assert_any_call(f"Command: {' '.join(error_cmd)}")
    mock_logging["error"].assert_any_call(f"Return code: {error_instance.returncode}")
    mock_logging["error"].assert_any_call(f"Stdout: {error_instance.output.strip()}")
    mock_logging["error"].assert_any_call(f"Stderr: {error_instance.stderr.strip()}")
#--------------------------------------------------------------------------------

#-------------------------- Test: FileNotFoundError --------------------------
def test_run_single_search_file_not_found_error(mocker, mock_paths, mock_logging):
    """Tests run_single_search when subprocess.run raises FileNotFoundError."""
    mock_subprocess_run = mocker.patch("subprocess.run")
    mock_subprocess_run.side_effect = FileNotFoundError("Script not found")

    clf_name = "MissingScriptClassifier"

    result = run_single_search(
        mock_paths["runner"],
        clf_name,
        mock_paths["dataset"],
        mock_paths["results"]
    )

    assert result is False
    mock_logging["error"].assert_any_call(f"Error: The script {mock_paths['runner']} was not found. Ensure the path is correct.")

def test_run_single_search_unexpected_error(mocker, mock_paths, mock_logging):
    """Tests run_single_search with a generic unexpected error during subprocess.run."""
    mock_subprocess_run = mocker.patch("subprocess.run")
    unexpected_exception = Exception("Unexpected boom!")
    mock_subprocess_run.side_effect = unexpected_exception

    clf_name = "UnexpectedFailClassifier"

    result = run_single_search(
        mock_paths["runner"],
        clf_name,
        mock_paths["dataset"],
        mock_paths["results"]
    )

    assert result is False
    mock_logging["error"].assert_any_call(
        f"An unexpected error occurred while trying to run {clf_name} on {mock_paths['dataset'].name}: {unexpected_exception}",
        exc_info=True
    )
#--------------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__])