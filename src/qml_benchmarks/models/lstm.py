import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
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

class LSTM(BaseEstimator, ClassifierMixin): #LSTM classifier
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

        if max_vmap is None:
            self.max_vmap = self.batch_size
        else:
            self.max_vmap = max_vmap

        self.params_ = None
        self.scaler = None

    def generate_key(self):
        return jax.random.PRNGKey(self.rng.integers(1000000))

    def initialize(self, n_features, classes=None):
        if classes is None:
            classes = [-1, 1]

        self.classes_ = classes
        self.n_classes_ = len(self.classes_)
        assert self.n_classes_ == 2
        assert 1 in self.classes_ and -1 in self.classes_
        self.lstm = construct_lstm(self.hidden_size)
        self.forward = self.lstm
        X0 = jnp.ones((1, self.seq_length, n_features))
        self.initialize_params(X0)

    def initialize_params(self, X):
        self.params_ = self.lstm.init(self.generate_key(), X)

    def fit(self, X, y):
        if X.ndim == 2:
            X = X[:, None, :]
        n_features = X.shape[-1]
        self.initialize(n_features, classes=np.unique(y))
        y = jnp.array(y, dtype=int)

        nsamples, seq_length, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        self.scaler = StandardScaler()
        self.scaler.fit(X_reshaped)
        X_scaled = self.scaler.transform(X_reshaped).reshape(nsamples, seq_length, n_features)
        X = jnp.array(X_scaled)

        def loss_fn(params, X, y):
            out = self.forward.apply(params, X)
            out = jnp.atleast_1d(out)
            if out.ndim > 1:
                logits = out.squeeze(axis=-1)
            else:
                logits = out
            y_positive = jax.nn.relu(y)
            loss = jnp.mean(optax.sigmoid_binary_cross_entropy(logits, y_positive))
            return loss

        if self.jit:
            loss_fn = jax.jit(loss_fn)
        optimizer = optax.adam
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
        predictions = self.predict_proba(X)
        mapped_predictions = np.argmax(predictions, axis=1)
        return np.take(self.classes_, mapped_predictions)

    def predict_proba(self, X):
        if X.ndim == 2:
            X = X[:, None, :]
        nsamples, seq_length, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        X_scaled = self.scaler.transform(X_reshaped).reshape(nsamples, seq_length, n_features)
        X = jnp.array(X_scaled)
        p1 = jax.nn.sigmoid(self.forward.apply(self.params_, X)[:, 0])
        predictions_2d = jnp.c_[1 - p1, p1]
        return predictions_2d

    def transform(self, X):
        if X.ndim == 2:
            X = X[:, None, :]
        nsamples, seq_length, n_features = X.shape
        X_reshaped = X.reshape(-1, n_features)
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X_reshaped)
        X_scaled = self.scaler.transform(X_reshaped).reshape(nsamples, seq_length, n_features)
        return jnp.array(X_scaled)
