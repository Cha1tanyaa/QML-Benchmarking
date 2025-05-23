import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

root = Path(__file__).resolve().parents[1]
src  = root / "src"
sys.path.insert(0, str(src))

# Import dataset generators from the data package
from qml_benchmarks.data.financial_times import generate_stock_features_and_labels
from qml_benchmarks.data.bars_and_stripes import generate_bars_and_stripes
from qml_benchmarks.data.hidden_manifold import generate_hidden_manifold_model
from qml_benchmarks.data.hyperplanes import generate_hyperplanes_parity
from qml_benchmarks.data.linearly_separable import generate_linearly_separable
from qml_benchmarks.data.two_curves import generate_two_curves

# Import all models from the models package
import qml_benchmarks.models as models

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
            print(f"Model {name} not found in models package.")
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
        print(f"Benchmarking model: {model_class.__name__}")
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
            print(f"Error using model {model_class.__name__}: {e}")
            results.append({
                "Model": model_class.__name__,
                "Accuracy": np.nan
            })
        
    return pd.DataFrame(results)

if __name__ == "__main__":
    # Map dataset names to their generator functions.
    # Adjust any generator-specific parameters if necessary.
    datasets = {
        "financial_times": lambda: generate_stock_features_and_labels(ticker="AAPL", start="2010-01-01", end="2024-01-01"),
        "bars_and_stripes": lambda: generate_bars_and_stripes(n_samples=300, n_features=4),
        "hidden_manifold": lambda: generate_hidden_manifold_model(n_samples=300, n_features=4, manifold_dimension=3),
        "hyperplanes": lambda: generate_hyperplanes_parity(n_samples=300, n_features=4, n_hyperplanes=5, dim_hyperplanes=2),
        "linearly_separable": lambda: generate_linearly_separable(n_samples=300, n_features=4),
        "two_curves": lambda: generate_two_curves(n_samples=300, n_features=4, degree=3, noise=0.1, offset=0.01),
    }
    
    all_results = []
    for ds_name, generator in datasets.items():
        try:
            X, y = generator()
            print(f"\nBenchmarking dataset: {ds_name}")
            df = benchmark_models(X, y)
            df["Dataset"] = ds_name
            all_results.append(df)
        except Exception as e:
            print(f"Could not generate dataset {ds_name}: {e}")
        
    if all_results:
        df_results = pd.concat(all_results, ignore_index=True)
        print("\nBenchmark Results:")
        print(df_results)
        
        # Save results to CSV
        results_path = os.path.join(os.getcwd(), "benchmark_results.csv")
        df_results.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
    else:
        print("No results to display.")