import os
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from importlib import import_module
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

root = Path(__file__).resolve().parents[1]
src  = root / "src"
sys.path.insert(0, str(src))

import qml_benchmarks.models as models

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

#------------------- Helper Functions -------------------
def get_models_list():
    """
    Returns a list of model classes from the models package.
    Adjust the list as needed.
    """
    model_classes = []
    for name in models.__all__:
        try:
            model_class = getattr(models, name)
            model_classes.append(model_class)
        except AttributeError:
            logging.warning(f"Model {name} not found in models package.")
    return model_classes

def benchmark_models(X, y, test_size=0.2, random_state=42):
    """
    Benchmarks all models on the provided dataset.
    Returns a DataFrame with the accuracy scores.
    """
    results = []
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    models_list = get_models_list()
    for model_class in models_list:
        logging.info(f"Benchmarking model: {model_class.__name__}")
        try:
            model = model_class()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            results.append({
                "Model": model_class.__name__,
                "Accuracy": acc
            })
        except Exception as e:
            logging.error(f"Error using model {model_class.__name__}: {e}")
            results.append({
                "Model": model_class.__name__,
                "Accuracy": np.nan
            })
    return pd.DataFrame(results)

def load_data_from_csv(file_path):
    """Loads X and y from a CSV file. Assumes target y is the last column."""
    data = pd.read_csv(file_path)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y
#------------------- End of Helper Functions -------------------

if __name__ == "__main__":
    csv_datasets_path = root / "paper_extension" / "datasets_generated"
    
    all_csv_results = []

    for csv_file in csv_datasets_path.rglob("*.csv"):
        ds_name = csv_file.stem
        try:
            logging.info(f"Loading dataset from: {csv_file}")
            X, y = load_data_from_csv(csv_file)
            if X.size == 0 or y.size == 0:
                logging.warning(f"No data loaded from {csv_file}. Skipping.")
                continue
            if X.shape[0] != y.shape[0]:
                logging.warning(f"Mismatch in samples vs labels in {csv_file}. X: {X.shape}, y: {y.shape}. Skipping.")
                continue
            logging.info(f"Benchmarking dataset: {ds_name}")
            df = benchmark_models(X, y)
            df["Dataset"] = ds_name
            all_csv_results.append(df)
        except Exception as e:
            logging.warning(f"Could not process dataset {ds_name} from {csv_file}: {e}")
        
    if all_csv_results:
        df_results = pd.concat(all_csv_results, ignore_index=True)
        logging.info("\nBenchmark Results from CSVs:")
        logging.info(df_results)
        results_output_path = Path(os.getcwd()) / "benchmark_results.csv"
        df_results.to_csv(results_output_path, index=False)
        logging.info(f"Results saved to {results_output_path}")
    else:
        logging.info("No CSV results to display or save.")