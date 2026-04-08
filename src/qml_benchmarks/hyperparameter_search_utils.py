# Copyright 2024 Xanadu Quantum Technologies Inc.
# Copyright 2025 Chaitanya Agrawal

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file has been modified by Chaitanya Agrawal from the original version.

"""Utility functions for hyperparameter search"""

import ast
import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


KNOWN_HYPERPARAMETER_HEADERS = {
    "hyperparameter",
    "parameter",
    "param",
    "name",
}
KNOWN_VALUE_HEADERS = {
    "best_value",
    "value",
    "val",
}


def read_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read data from a csv file where each row is a data sample.
    The columns are the input features and the last column specifies a label.

    Return a 2-d array of inputs and an array of labels, X,y.

    Args:
        path (str | Path): path to data
    """
    # The data is stored on a CSV file with the last column being the label
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")

    data = pd.read_csv(csv_path, header=None)
    if data.shape[1] < 2:
        raise ValueError(f"Dataset must contain at least one feature column and one label column: {csv_path}")

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


def construct_hyperparameter_grid(
    hyperparameter_settings: dict[str, dict[str, dict[str, Any]]],
    classifier_name: str,
) -> dict[str, np.ndarray | list[Any]]:
    """Constructs a grid of hyperparameters from the dictionary of hyperparameter
    settings for a given classifier.

    Args:
        hyperparameter_settings (dict): a dictionary of hyperparameter settings
        classifier_name (str): classifier name

    Returns:
        hyperparameter_grid (dict): A grid of hyperparameters to search
    """
    if classifier_name not in hyperparameter_settings:
        raise KeyError(f"No hyperparameter settings found for classifier '{classifier_name}'")

    hyperparams = hyperparameter_settings[classifier_name].keys()
    hyperparameter_grid = {}

    for hyperparam in hyperparams:
        if hyperparameter_settings[classifier_name][hyperparam]["type"] == "list":
            val = hyperparameter_settings[classifier_name][hyperparam]["val"]
            dtype = hyperparameter_settings[classifier_name][hyperparam]["dtype"]
            if dtype == "tuple":
                hyperparameter_grid[hyperparam] = [ast.literal_eval(v) for v in val]
            else:
                hyperparameter_grid[hyperparam] = np.array(val, dtype=dtype)

    return hyperparameter_grid


def _parse_csv_value(value: Any) -> Any:
    """Parse scalar and literal values from CSV text without executing code."""
    if not isinstance(value, str):
        return value

    value = value.strip()
    if value == "":
        return value

    lowered = value.lower()

    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "none":
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        pass

    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def parse_hyperparameters(values: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize a hyperparameter dictionary parsed from a CSV source."""
    if values is None:
        return {}

    parsed: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, str):
            parsed[key] = _parse_csv_value(value)
        elif isinstance(value, np.integer):
            parsed[key] = int(value)
        elif isinstance(value, np.floating):
            parsed[key] = int(value) if float(value).is_integer() else float(value)
        elif isinstance(value, float) and value.is_integer():
            parsed[key] = int(value)
        else:
            parsed[key] = value
    return parsed


def _is_header_row(row: list[str]) -> bool:
    if len(row) < 2:
        return False

    first = row[0].strip().lower()
    second = row[1].strip().lower()
    return first in KNOWN_HYPERPARAMETER_HEADERS and second in KNOWN_VALUE_HEADERS


def csv_to_dict(file_path: str | Path) -> dict[str, Any]:
    """Read a csv file and interpret the content as a dictionary.

    Args:
        file_path (str | Path): path to csv file
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Hyperparameter file not found: {path}")

    parsed_values: dict[str, Any] = {}
    with path.open("r", newline="", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        first_row = next(csvreader, None)
        if first_row is not None and not _is_header_row(first_row):
            rows_iterable = [first_row, *csvreader]
        else:
            rows_iterable = csvreader

        for row in rows_iterable:
            if len(row) < 2:
                continue
            hyperparameter = row[0].strip()
            if not hyperparameter:
                continue

            # Some generated CSV rows can include unquoted commas in values.
            value = ",".join(row[1:]).strip()
            parsed_values[hyperparameter] = _parse_csv_value(value)

    return parsed_values
