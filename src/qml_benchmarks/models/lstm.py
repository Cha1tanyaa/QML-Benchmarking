import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, RegressorMixin
from qml_benchmarks.model_utils import train

jax.config.update("jax_enable_x64", True)

def construct_lstm(hidden_size, num_layers=1):
    class LSTMModel(nn.Module):
        @nn.compact
        def __call__(self, x):
            batch_size, seq_length, feature_dim = x.shape
            h0 = jnp.zeros((batch_size, hidden_size), dtype=x.dtype)
            c0 = jnp.zeros((batch_size, hidden_size), dtype=x.dtype)
            carry = (h0, c0)
            lstm_cell = nn.LSTMCell(features=hidden_size)
            for t in range(seq_length):
                carry, _ = lstm_cell(carry, x[:, t, :])
            output = nn.Dense(features=1)(carry[1])
            return output
    return LSTMModel()

class LSTM(BaseEstimator, RegressorMixin): #LSTM classifier
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
        scaling=1.0,
    ):
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.scaling = scaling
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
        self.lstm = construct_lstm(self.hidden_size, self.seq_length)
        self.forward = self.lstm
        X0 = jnp.ones((1, self.seq_length, n_features))
        self.initialize_params(X0)

    def initialize_params(self, X):
        self.params_ = self.lstm.init(self.generate_key(), X)

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
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X_reshaped)
        X_scaled = self.scaler.transform(X_reshaped).reshape(nsamples, seq_length, n_features)
        return jnp.array(X_scaled)
    