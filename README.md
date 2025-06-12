# Benchmarking for quantum machine learning models

This repository contains a toolkit to compare the performance of near-term quantum machine learning (QML)
with standard classical machine learning models on supervised learning tasks. 

It is based on pipelines using [Pennylane](https://pennylane.ai/) for the simulation of quantum circuits, 
[JAX](https://jax.readthedocs.io/en/latest/index.html) for training, 
and [scikit-learn](https://scikit-learn.org/) for the benchmarking workflows. 

Version 0.1 of the code can be used to reproduce the results in the study "Better than classical? The subtle art of benchmarking quantum machine learning models".

## Original Work and Contributions

This repository builds upon the framework and tools originally developed by XANADU for their study "Better than classical? The subtle art of benchmarking quantum machine learning models". The original code (Version 0.1) provides the foundation for quantum circuit & classical simulation, training, and benchmarking workflows.

The following modifications and additions have been made to the original repository:

*   **New Model Implementations:** Introduced several classical and quantum-inspired models, including Feedforward Neural Networks (FFN), Support Vector Machines (SVM), XGBoost, Long Short-Term Memory (LSTM) networks, and Quantum LSTMs (QLSTM).
*   **Real-World Dataset Integration:** Incorporated new real-world datasets for benchmarking, specifically focusing on Financial Times series data and Credit Card Fraud detection.
*   **Enhanced Automation:** Developed automated pipelines for:
    *   Extended hyperparameter search.
    *   Systematic dataset generation.
    *   Comprehensive evaluation procedures.

*   The `paper_extension` folder has been added to house all new benchmarks, datasets, generated data and results from the extended research. This includes:
    *   `paper_extension/benchmarks`: Scripts for data generation, hyperparameter search & extended benchmarking
    *   `paper_extension/results_phase1`, `paper_extension/results_phase2`, `paper_extension/results_phase3`: Results of the extended research
*   The `src\qml_benchmarks\models` folder includes the newly introduced models
*   The `src\qml_benchmarks\models` folder includes the newly added datasets

The core functionalities for defining quantum and classical models, as well as data generation, largely leverage the original `qml_benchmarks` package, with specific modifications documented within the commit history.

## Overview

A short summary of the various folders in this repository is as follows:
- `paper`: contains code and results to reproduce the results in the original paper provided by XANADU
  - `benchmarks`: scripts that generate datasets of varied difficulty and feature dimensions for the original study
  - `plots`: scripts that generate the plots and additional experiments in the original paper
  - `results`: data files recording the results of the benchmark experiments that the original study is based on
- `paper_extension`: contains all materials related to the extended research building upon the original XANADU framework
  - `benchmarks`: Scripts to generate datasets, run hyperparameter search and benchmark the models
  - `results_phase1`, `results_phase2`, `results_phase3`: Data files from different phases of the extended research
- `scripts`: example code for how to benchmark a model on a dataset 
- `src/qml_benchmarks`: a simple Python package defining quantum and classical models, 
   as well as data generating functions

## Installation

You can install the `qml_benchmarks` package in your environment with

```bash
pip install -e .
```

from the root directory of the repository. This will install the package in
editable mode, meaning that changes to the code will be reflected in the
installed package.

Dependencies of this package can be installed in your environment by running 

```bash
pip install -r requirements.txt
```

## Adding a custom model

We use the [Scikit-learn API](https://scikit-learn.org/stable/developers/develop.html) to create 
models and perform hyperparameter search.

A minimal template for a new quantum model is as follows, and can be stored 
in `src/qml_benchmarks/models/my_model.py`:

```python
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin


class MyModel(BaseEstimator, ClassifierMixin):
    def __init__(self, hyperparam1="some_value",  random_state=42):

        # store hyperparameters as attributes
        self.hyperparam1 = hyperparam1
                    
        # reproducibility is ensured by creating a numpy PRNG and using it for all
        # subsequent random functions. 
        self._random_state = random_state
        self._rng = np.random.default_rng(random_state)
            
        # define data-dependent attributes
        self.params_ = None
        self.n_qubits_ = None

    def fit(self, X, y):
        """Fit the model to data X and labels y.

        Add your custom training loop here and store the trained model parameters in `self.params_`.
        Set the data-dependent attributes, such as `self.n_qubits_`.
        
        Args:
            X (array_like): Data of shape (n_samples, n_features)
            y (array_like): Labels of shape (n_samples,)
        """
        # ... your code here ...        
        

    def predict(self, X):
        """Predict labels for data X.
        
        Args:
            X (array_like): Data of shape (n_samples, n_features)
        
        Returns:
            array_like: Predicted labels of shape (n_samples,)
        """
        # ... your code here ...
        
        return y_pred

    def predict_proba(self, X):
        """Predict label probabilities for data X.

        Args:
            X (array_like): Data of shape (n_samples, n_features)

        Returns:
            array_like: Predicted label probabilities of shape
            (n_samples, n_classes)
        """
        # ... your code here ...
        return y_pred_proba
```

To ensure compatibility with scikit-learn functionalities, all models should
inherit the `BaseEstimator` and `ClassifierMixin` classes.  Implementing the `fit`,
`predict`, and `predict_proba` methods is sufficient. 

The model parameters are stored as a dictionary in `self.params_`. 

There are two types of other attributes: those initialized when the instance of the class is 
created, and those that are only known when data is seen (for example, the number of qubits 
may depend on the dimension of input vectors). In the latter case, a default (i.e., `self.n_qubits_ = None`) 
is set in the `init` function, and the value is typically updated when `fit` is called for the first time.

It can be useful to implement an `initialize` method which initializes an untrained model with random 
parameters so that `predict_proba` and `predict` can be called. 

The custom model can be used as follows:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from qml_benchmarks.models.my_model import MyModel

# load data and use labels -1, 1
X, y = make_classification(n_samples=100, n_features=2,
                           n_informative=2, n_redundant=0, random_state=42)
y = np.array([-1 if y_ == 0 else 1 for y_ in y])

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# fit model
model = MyModel(hyperparam1=0.5)
model.fit(X_train, y_train)

# score the model
print(model.score(X_test, y_test))
```

## Datasets

The `src/qml_benchmarks/data` module provides generating functions to create datasets. These are leveraged in `paper_extension/benchmarks/generate_datasets.py` to generate datasets.

A generating function can be used like this:

```
python paper_extension/benchmarks/generate_datasets.py
```

The parameters can be adjusted inside the file under DEFAULT_PARAMS. Note that some datasets might have different return data structures, for example if the train/test split is changed.

This will create a new folder in `paper_extension/datasets_generated` containing the datasets.

## Running hyperparameter optimization

In the folder `paper_extension` we provide scripts through which an automatic Hyperparameter Search is triggered. In order to conduct our study we created two different scripts.

The first script is able to run all models on all given datasets and can be run with the command
```
python paper_extension/benchmarks/automatic_hyperparmeter_search.py
```

The second script can be used to reproduce the results of our paper
```
python paper_extension/benchmarks/extension_hyperparameter_search.py
```

Note that if newly added models and datasets are not used for binary classification or regression the scripts have to be adjusted accordingly

Unless otherwise specified, the hyperparameter grid is loaded from `src/qml_benchmarks/hyperparameter_settings.py`.
One can override the default grid of hyperparameters by specifying the hyperparameter list,
where the datatype is inferred from the default values.

The script creates two CSV files that contains the detailed results of hyperparameter search and the best 
hyperparameters obtained in the search. These files are similar to the ones stored in the `paper/results`
folder for the original study, and are stored in `paper_extension/results_phase*` folders for the extended study.

## Benchmarking Models

The folder `paper_extension` contains scripts to automatically Benchmark the all models on which hyperparameter search was conducted. It automatically retrieves the hyperparameter results, the datasets and the models given that the previous instructions were followed. 

Note that since we extended the research we also included regression tasks and since those demand different metrics we created two different scripts.

The first script is able to conduct benchmarking on all Binary classification tasks and can be run with the command
```
python paper_extension/benchmarks/benchmark_pipeline_classifier.py
```

The first script is able to conduct benchmarking on all Binary classification tasks and can be run with the command
```
python paper_extension/benchmarks/benchmark_pipeline_regressor.py
```

The results are than saved in the hyperparameter search result folders in `paper_extension/results_phase*`. 

## Feedback 

Please help us improve this repository and report problems by opening an issue or pull request.