# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module containing models to be used in benchmarks."""

from qml_benchmarks.models.circuit_centric import CircuitCentricClassifier
from qml_benchmarks.models.convolutional_neural_network import ConvolutionalNeuralNetwork

from qml_benchmarks.models.data_reuploading import (
    DataReuploadingClassifier,
    DataReuploadingClassifierNoScaling,
    DataReuploadingClassifierNoCost,
    DataReuploadingClassifierNoTrainableEmbedding,
    DataReuploadingClassifierSeparable,
)
from qml_benchmarks.models.dressed_quantum_circuit import (
    DressedQuantumCircuitClassifier,
    DressedQuantumCircuitClassifierOnlyNN,
    DressedQuantumCircuitClassifierSeparable,
)

from qml_benchmarks.models.iqp_kernel import IQPKernelClassifier
from qml_benchmarks.models.iqp_variational import IQPVariationalClassifier
from qml_benchmarks.models.projected_quantum_kernel import ProjectedQuantumKernel
from qml_benchmarks.models.quantum_boltzmann_machine import (
    QuantumBoltzmannMachine,
    QuantumBoltzmannMachineSeparable
)
from qml_benchmarks.models.quantum_kitchen_sinks import QuantumKitchenSinks
from qml_benchmarks.models.quantum_metric_learning import QuantumMetricLearner
from qml_benchmarks.models.quanvolutional_neural_network import QuanvolutionalNeuralNetwork
from qml_benchmarks.models.separable import (
    SeparableVariationalClassifier,
    SeparableKernelClassifier,
)
from qml_benchmarks.models.tree_tensor import TreeTensorClassifier
from qml_benchmarks.models.vanilla_qnn import VanillaQNN
from qml_benchmarks.models.weinet import WeiNet

from sklearn.svm import SVC as SVC_base

from qml_benchmarks.models.qlstm import QLSTM
from qml_benchmarks.models.lstm import LSTM
from qml_benchmarks.models.mlp import MLP
from qml_benchmarks.models.xgboost import XGBoost
from qml_benchmarks.models.svm import SVM
__all__ = [
    "CircuitCentricClassifier",
    "ConvolutionalNeuralNetwork",
    "DataReuploadingClassifier",
    "DataReuploadingClassifierNoScaling",
    "DataReuploadingClassifierNoCost",
    "DataReuploadingClassifierNoTrainableEmbedding",
    "DataReuploadingClassifierSeparable",
    "DressedQuantumCircuitClassifier",
    "DressedQuantumCircuitClassifierOnlyNN",
    "DressedQuantumCircuitClassifierSeparable",
    "IQPKernelClassifier",
    "IQPVariationalClassifier",
    "QLSTM",
    "LSTM",
    "MLP",
    "XGBoost",
    "SVM",
    "ProjectedQuantumKernel",
    "QuantumBoltzmannMachine",
    "QuantumBoltzmannMachineSeparable",
    "QuantumKitchenSinks",
    "QuantumMetricLearner",
    "QuanvolutionalNeuralNetwork",
    "SeparableVariationalClassifier",
    "SeparableKernelClassifier",
    "TreeTensorClassifier",
    "VanillaQNN",
    "WeiNet",
    "SVC",
]

class SVC(SVC_base):
    def __init__(
            self,
            C=1.0,
            degree=3,
            gamma="scale",
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=0.001,
            max_iter=-1,
            random_state=None,
    ):
        super().__init__(
            C=C,
            kernel="rbf",
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            max_iter=max_iter,
            random_state=random_state,
        )
