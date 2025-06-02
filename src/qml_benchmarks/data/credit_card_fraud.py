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