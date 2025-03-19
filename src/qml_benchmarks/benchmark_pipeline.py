import os
import sys
import numpy as np
import pandas as pd
from importlib import import_module
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import dataset generators from the data package
from data.financial_times import generate_stock_features_and_labels
from data.bars_and_stripes import generate_bars_and_stripes
from data.hidden_manifold import generate_hidden_manifold_model
from data.hyperplanes import generate_hyperplanes_parity
from data.linearly_separable import generate_linearly_separable
from data.two_curves import generate_two_curves

# Import all models from the models package
import models as models

def get_models_list():
    model_classes = []
    for name in models.__all__:
        if name == "SeparableKernelClassifier":
            continue
        try:
            model_class = getattr(models, name)
            model_classes.append(model_class)
        except AttributeError:
            print(f"Model {name} not found in models package.")
    return model_classes

def benchmark_models(X, y, test_size=0.2, random_state=42):   #returns a dataframe with model names and their accuracy scores
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

if __name__ == "__main__":              #map the datasets to their generator functions and benchmark the models
    datasets = {
        "financial_times": lambda: generate_stock_features_and_labels(ticker="AAPL", start="2010-01-01", end="2024-01-01"),
        "bars_and_stripes": lambda: generate_bars_and_stripes(n_samples=300, height=4, width=4, noise_std=0.1),
        "hidden_manifold": lambda: generate_hidden_manifold_model(n_samples=300, n_features=4, manifold_dimension=3),
        "hyperplanes": lambda: generate_hyperplanes_parity(n_samples=300, n_features=4, n_hyperplanes=5, dim_hyperplanes=2),
        "linearly_separable": lambda: generate_linearly_separable(n_samples=300, n_features=4, margin=0.1),
        "two_curves": lambda: generate_two_curves(n_samples=300, n_features=4, degree=3, noise=0.1, offset=0.01),
    }
    
    all_results = []
    for ds_name, generator in datasets.items():             #benchmark all models on each dataset
        print('cpu count:' + str(os.cpu_count()))
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
        if 'info' in df_results.columns:
            split_cols = df_results['info'].str.split(',', expand=True)
            split_cols.columns = [f'info_{i+1}' for i in range(split_cols.shape[1])]
            df_results = df_results.drop(columns=['info']).join(split_cols)
        print("\nBenchmark Results:")
        print(df_results)
        
        # Save results to CSV
        results_path = os.path.join(os.getcwd(), "benchmark_results.csv")
        df_results.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")
    else:
        print("No results to display.")