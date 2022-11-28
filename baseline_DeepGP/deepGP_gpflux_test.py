'''
https://secondmind-labs.github.io/GPflux/notebooks/gpflux_features.html
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


tf.keras.backend.set_floatx("float64")  # we want to carry out GP calculations in 64 bit
tf.get_logger().setLevel("INFO")

import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import synthetic_regression_problem
from utils import standardise_labels


X, Y, func = synthetic_regression_problem(train_len=800, noise_level=0.1)
Xtest = np.linspace(0,1,1000).reshape(-1,1)
Ytest = func(Xtest)
plt.plot(X, Y, "kx")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

import gpflux

from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from gpflux.models import DeepGP

config = Config(
    num_inducing=100, inner_layer_qsqrt_factor=1e-5, likelihood_noise_variance=1e-4, whiten=True
)
deep_gp: DeepGP = build_constant_input_dim_deep_gp(X, num_layers=2, config=config)


# From the `DeepGP` model we instantiate a training model which is a `tf.keras.Model`
training_model: tf.keras.Model = deep_gp.as_training_model()

# Following the Keras procedure we need to compile and pass a optimizer,
# before fitting the model to data
training_model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.01))

callbacks = [
    # Create callback that reduces the learning rate every time the ELBO plateaus
    tf.keras.callbacks.ReduceLROnPlateau("loss", factor=0.95, patience=3, min_lr=1e-6, verbose=0),
    # Create a callback that writes logs (e.g., hyperparameters, KLs, etc.) to TensorBoard
    gpflux.callbacks.TensorBoard(),
    # Create a callback that saves the model's weights
    tf.keras.callbacks.ModelCheckpoint(filepath="ckpts/", save_weights_only=True, verbose=0),
]

history = training_model.fit(
    {"inputs": X, "targets": Y},
    batch_size=12,
    epochs=200,
    callbacks=callbacks,
    verbose=0,
)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 3))
ax1.plot(history.history["loss"])
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Objective = neg. ELBO")

ax2.plot(history.history["lr"])
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Learning rate")
plt.show()


def plot(model, X, Y, Xtest, ax=None):
    if ax is None:
        fig, ax = plt.subplots()


    out = model(Xtest)

    mu = out.f_mean.numpy().squeeze()
    var = out.f_var.numpy().squeeze()
    Xtest = Xtest.squeeze()
    lower = mu - 2 * np.sqrt(var)
    upper = mu + 2 * np.sqrt(var)

    ax.set_ylim(Y.min() - 0.5, Y.max() + 0.5)
    ax.plot(X, Y, "kx", alpha=0.5)
    ax.plot(Xtest, mu, "C1")

    ax.fill_between(Xtest, lower, upper, color="C1", alpha=0.3)


prediction_model = deep_gp.as_prediction_model()
plot(prediction_model, X, Y, Xtest)
plt.show()