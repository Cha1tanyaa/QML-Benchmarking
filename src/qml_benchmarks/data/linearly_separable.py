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

import numpy as np


def generate_linearly_separable(n_samples, n_features, margin):
    """Data generation procedure for 'linearly separable'.

    Args:
        n_samples (int): number of samples to generate
        n_features (int): dimension of the data samples
        margin (float): width between hyperplane and closest samples
    """

    w_true = np.ones(n_features)

    # Sample a larger candidate set because margin filtering can discard many points.
    X = 2 * np.random.rand(2 * n_samples, n_features) - 1

    # only retain data outside a margin
    X = [x for x in X if np.abs(np.dot(x, w_true)) > margin]
    X = X[:n_samples]
    if len(X) < n_samples:
        raise ValueError(
            f"Could not generate {n_samples} samples with margin={margin}. "
            f"Only {len(X)} samples remained after filtering."
        )

    y = [np.dot(x, w_true) for x in X]
    y = [-1 if y_ > 0 else 1 for y_ in y]
    return np.asarray(X), np.asarray(y)
