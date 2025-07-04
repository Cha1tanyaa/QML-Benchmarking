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

"""Module containing data generating functions for classification tasks."""

from qml_benchmarks.data.bars_and_stripes import generate_bars_and_stripes
from qml_benchmarks.data.hidden_manifold import generate_hidden_manifold_model
from qml_benchmarks.data.hyperplanes import generate_hyperplanes_parity
from qml_benchmarks.data.linearly_separable import generate_linearly_separable
from qml_benchmarks.data.two_curves import generate_two_curves
from qml_benchmarks.data.financial_times import generate_stock_features_and_labels
from qml_benchmarks.data.credit_card_fraud import generate_credit_card_fraud_features_and_labels