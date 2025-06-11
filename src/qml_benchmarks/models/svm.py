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

import numpy as np
import jax
import jax.numpy as jnp
import optax
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from qml_benchmarks.model_utils import train

jax.config.update("jax_enable_x64", True)

class SVM(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(
        self,
        learning_rate=1.0,
        batch_size=32,
        max_steps=1000,
        convergence_interval=100,
        random_state=42,
        scaling=1.0,
        jit=True,
        max_vmap=None,
    ):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_steps = max_steps
        self.convergence_interval = convergence_interval
        self.random_state = random_state
        self.scaling = scaling
        self.jit = jit
        self.rng = np.random.default_rng(random_state)
        if max_vmap is None:
            self.max_vmap = self.batch_size
        else:
            self.max_vmap = max_vmap
        self.scaler_ = None
        self.params_ = None
        self.n_features_ = None
        self.classes_ = None

    def generate_key(self):
        return jax.random.PRNGKey(int(self.rng.integers(1000000)))

    def initialize(self, n_features, classes=None):
        self.n_features_ = n_features
        self.classes_ = classes
        self.scaler_ = StandardScaler()
        return self

    def initialize_params(self):
        key = self.generate_key()
        w = jax.random.normal(key, (self.n_features_,)) * 0.01
        b = jnp.zeros(())
        self.params_ = {"w": w, "b": b}

    def construct_model(self):
        def decision(params, X):
            return jnp.dot(X, params["w"]) + params["b"]
        self.decision_function = decision

    def fit(self, X, y):
        self.initialize(X.shape[1], classes=np.unique(y))
        X = self.scaler_.fit_transform(X) * self.scaling
        y = jnp.where(y == self.classes_[0], -1, 1)
        self.initialize_params()
        self.construct_model()
        optimizer = optax.adam

        def loss_fn(params, Xb, yb):
            margins = yb * self.decision_function(params, Xb)
            hinge = jnp.maximum(0.0, 1.0 - margins)
            return jnp.mean(hinge)

        if self.jit:
            loss_fn = jax.jit(loss_fn)

        self.params_ = train(
            self,
            loss_fn,
            optimizer,
            X,
            y,
            self.generate_key,
            convergence_interval=self.convergence_interval,
        )
        return self

    def predict(self, X):
        X = self.scaler_.transform(X) * self.scaling
        scores = self.decision_function(self.params_, X)
        return np.where(scores >= 0, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        X = self.scaler_.transform(X) * self.scaling
        scores = self.decision_function(self.params_, X)
        prob_pos = jax.nn.sigmoid(scores)
        probs = jnp.stack([1 - prob_pos, prob_pos], axis=1)
        return np.array(probs)