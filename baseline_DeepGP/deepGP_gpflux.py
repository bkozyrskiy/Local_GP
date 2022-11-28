'''
https://secondmind-labs.github.io/GPflux/notebooks/gpflux_features.html
'''


import gpflux
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

from gpflux.architectures import Config, build_constant_input_dim_deep_gp
from gpflux.models import DeepGP
import tensorflow as tf
tf.keras.backend.set_floatx("float64")  # we want to carry out GP calculations in 64 bit
tf.get_logger().setLevel("INFO")

import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset
from utils import standardise_labels

if __name__ == '__main__':
    dataset = 'yacht'
    train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=True)
    train_x, train_y = train_dataset.tensors[0].numpy(), train_dataset.tensors[1].squeeze().numpy()
    test_x, test_y = test_dataset.tensors[0].numpy(), test_dataset.tensors[1].squeeze().numpy()
    
    # scaler = MinMaxScaler()
    # scaler.fit(train_x)
    # train_x = scaler.transform(train_x)
    # test_x = scaler.fit_transform(test_x)
    
    # train_y, mean, std = standardise_labels(train_y)
    mean=0
    std=1
    
    
    config = Config(
        num_inducing=100, inner_layer_qsqrt_factor=1e-5, likelihood_noise_variance=1e-3, whiten=True
    )
    deep_gp: DeepGP = build_constant_input_dim_deep_gp(train_x, num_layers=2, config=config)
    training_model: tf.keras.Model = deep_gp.as_training_model()
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
        {"inputs": train_x, "targets": train_y},
        batch_size=None,
        epochs=400,
        callbacks=callbacks,
        verbose=0,
    )
    plt.plot(history.history['loss'])
    plt.show()
    prediction_model = deep_gp.as_prediction_model()
    
    train_out = prediction_model(train_x).f_mean.numpy()
    print(f'Train MSE {((train_out * std - train_y * std)**2).mean()}')
    test_out = prediction_model(test_x).f_mean.numpy()
    print(f'Test MSE {((test_out * std + mean - test_y)**2).mean()}')
    