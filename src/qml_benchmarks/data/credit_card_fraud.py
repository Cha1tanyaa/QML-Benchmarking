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

import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple
import numpy as np
from pathlib import Path

def generate_credit_card_fraud_features_and_labels(
    file_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads and preprocesses the Credit Card Fraud Detection dataset from a local CSV file.

    Args:
        file_path (Path): The path to the local CSV file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the feature matrix (X)
                                       and the target vector (y).
    """
    df = pd.read_csv(
        file_path,
        encoding=None,
        sep=",",
        engine="python",
        on_bad_lines="skip"
    )

    X_df = df.drop('Class', axis=1)
    y_series = df['Class']

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_df)

    return X_scaled, y_series.values