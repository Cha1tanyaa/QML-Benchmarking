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
import flax.linen as nn
import pennylane as qml
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from qml_benchmarks.model_utils import train

jax.config.update("jax_enable_x64", True)

class QuantumLayer(nn.Module):
    hidden_dim: int

    def setup(self):
        self.n_qubits = min(self.hidden_dim, 16)
        self.dev = qml.device("default.qubit", wires=self.n_qubits)

        @qml.qnode(self.dev, interface="jax", diff_method="best")
        def quantum_circuit(inputs, weights):
            """
            Quantum circuit.
            inputs: scaled hidden state from LSTM, shape (n_qubits,)
            weights: trainable parameters for the quantum gates, shape (n_qubits,)
            """
            for i in range(self.n_qubits):
                qml.RY(inputs[i], wires=i)
                qml.RX(weights[i], wires=i)
            return qml.expval(qml.PauliZ(0))

        self.qc = quantum_circuit
        self.vmap_qc = jax.vmap(self.qc, in_axes=(0, None), out_axes=0)

    @nn.compact
    def __call__(self, x):
        theta = self.param("theta", nn.initializers.normal(), (self.hidden_dim,))
        scaled_x = x * jnp.pi
        out = self.vmap_qc(scaled_x, theta)
        out = jnp.asarray(out, dtype=x.dtype)
        out = out[:, None]
        out = nn.Dense(features=self.hidden_dim, kernel_init=nn.initializers.zeros)(out)
        return out

class QLSTMCell(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, carry, x):
        lstm_cell = nn.LSTMCell(features=self.hidden_size)
        new_carry, y = lstm_cell(carry, x)
        qlayer = QuantumLayer(hidden_dim=self.hidden_size)
        new_hidden = qlayer(new_carry[1])
        new_carry = (new_carry[0], new_hidden)
        return new_carry, new_hidden

def construct_qlstm(hidden_size, seq_length):
    class QLSTMModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            batch_size, seq_length, feature_dim = x.shape
            h0 = jnp.zeros((batch_size, hidden_size), dtype=x.dtype)
            c0 = jnp.zeros((batch_size, hidden_size), dtype=x.dtype)
            carry = (h0, c0)
            qlstm_cell = QLSTMCell(hidden_size=hidden_size)
            for t in range(seq_length):
                carry, _ = qlstm_cell(carry, x[:, t, :])
            output = nn.Dense(features=1)(carry[1])
            return output
    return QLSTMModel()

class QLSTM(BaseEstimator, RegressorMixin): #Quantum LSTM classifier
    def __init__(
        self,
        hidden_size=128,
        seq_length=10,
        learning_rate=0.001,
        convergence_interval=200,
        max_steps=10000,
        batch_size=32,
        max_vmap=None,
        jit=True,
        random_state=42,
    ):
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.jit = jit
        self.convergence_interval = convergence_interval
        self.batch_size = batch_size
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)

        self._init_max_vmap_arg = max_vmap

        if self._init_max_vmap_arg is None:
            self.max_vmap = self.batch_size
        else:
            self.max_vmap = self._init_max_vmap_arg
        
        self.params_ = None
        self.scaler = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def initialize(self, n_features):
        print(f"QLSTM.initialize called with hidden_size: {self.hidden_size}")
        self.qlstm = construct_qlstm(self.hidden_size, self.seq_length)
        self.forward = self.qlstm
        X0 = jnp.ones((1, self.seq_length, n_features))
        self.initialize_params(X0)

    def initialize_params(self, X):
        self.params_ = self.qlstm.init(self.generate_key(), X)

    def fit(self, X, y):
        if self._init_max_vmap_arg is None:
            self.max_vmap = self.batch_size
        if X.ndim == 2:
            X = X[:, np.newaxis, :] 
        n_features = X.shape[-1]
        self.initialize(n_features)
        y = jnp.array(y, dtype=jnp.float64)

        nsamples, seq_length, n_features = X.shape
        if seq_length != self.seq_length:
            pass
        X_reshaped = X.reshape(-1, n_features)
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X_reshaped)
        X_scaled = self.scaler.transform(X_reshaped).reshape(nsamples, seq_length, n_features)
        X_jax = jnp.array(X_scaled)
        y_jax = y

        def loss_fn(params, X_batch, y_batch):
            predictions = self.forward.apply(params, X_batch)
            loss = jnp.mean(optax.squared_error(predictions.squeeze(-1), y_batch))
            return loss

        if self.jit:
            loss_fn = jax.jit(loss_fn)
        optimizer = optax.adam
        self.params_ = train(
            self,
            loss_fn,
            optimizer,
            X_jax,
            y_jax,
            self.generate_key,
            convergence_interval=self.convergence_interval,
        )
        return self

    def predict(self, X):
        if X.ndim == 2:
             X = X[:, np.newaxis, :]
        nsamples, seq_length, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped).reshape(nsamples, seq_length, n_features)
        X_jax = jnp.array(X_scaled)
        predictions = self.forward.apply(self.params_, X_jax)
        return np.array(predictions.squeeze(-1))

    def transform(self, X):     
        if X.ndim == 2:
            X = X[:, np.newaxis, :]
        nsamples, seq_length, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped).reshape(nsamples, seq_length, n_features)
        return jnp.array(X_scaled)
