import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from qml_benchmarks.model_utils import train

jax.config.update("jax_enable_x64", True)

def get_activation_function(name: str):
    if name == "relu":
        return nn.relu
    elif name == "tanh":
        return nn.tanh
    elif name == "sigmoid":
        return nn.sigmoid
    else:
        raise ValueError(f"Unsupported activation function: '{name}'")

def construct_ffn(hidden_layers, activation_name: str):
    selected_activation_fn = get_activation_function(activation_name)

    class FeedforwardNN(nn.Module):
        @nn.compact
        def __call__(self, x):
            for size in hidden_layers:
                x = nn.Dense(features=size)(x)
                x = selected_activation_fn(x)
            x = nn.Dense(features=1)(x)  # Single output neuron
            return x

    return FeedforwardNN()
   

class MLP(BaseEstimator, ClassifierMixin): #Feedforward neural network
    def __init__(
        self,
        hidden_layer_sizes=[128, 64],
        activation="relu",
        learning_rate=0.01,
        alpha=0.001,
        convergence_interval=200,
        max_steps=2000,
        batch_size=32,
        max_vmap=None,
        jit=True,
        random_state=42,
        scaling=1.0,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.alpha = alpha
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
        #assert 1 in self.classes_ and -1 in self.classes_

        self.ffn = construct_ffn(self.hidden_layer_sizes, self.activation)
        self.forward = self.ffn

        X0 = jnp.ones(shape=(1, n_features))
        self.initialize_params(X0)

    def initialize_params(self, X):
        self.params_ = self.ffn.init(self.generate_key(), X)

    def fit(self, X, y):

        self.initialize(X.shape[1], classes=np.unique(y))

        y = jnp.array(y, dtype=int)

        # scale input data
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        X = self.transform(X)

        def loss_fn(params, X, y):
            y = jax.nn.relu(y)  
            vals = self.forward.apply(params, X)[:, 0]
            loss = jnp.mean(optax.sigmoid_binary_cross_entropy(vals, y))
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
        X = self.transform(X)
        p1 = jax.nn.sigmoid(self.forward.apply(self.params_, X)[:, 0])
        predictions_2d = jnp.c_[1 - p1, p1]
        return predictions_2d

    def transform(self, X):
        if self.scaler is None:
            self.scaler = StandardScaler()
            self.scaler.fit(X)

        X = self.scaler.transform(X) * self.scaling
        return jnp.array(X)
