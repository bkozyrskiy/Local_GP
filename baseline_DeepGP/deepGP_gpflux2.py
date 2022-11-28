'''
https://secondmind-labs.github.io/GPflux/index.html#id7
'''

import numpy as np
import gpflux
import gpflow
import tensorflow as tf
from scipy.cluster.vq import kmeans2
from gpflux.helpers import construct_basic_inducing_variables, construct_basic_kernel
tf.keras.backend.set_floatx("float64") 

import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset
from utils import standardise_labels

if __name__ == '__main__':
    
    dataset = 'yacht'
    train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=True)
    X, Y = train_dataset.tensors[0].numpy(), train_dataset.tensors[1].squeeze().numpy()
    test_x, test_y = test_dataset.tensors[0].numpy(), test_dataset.tensors[1].squeeze().numpy()
    
    
    num_inducing = 100
    Z, _ = kmeans2(X, k=num_inducing, minit="points")
    
    
    # Layer 1
    
    kernel1 = construct_basic_kernel(
        gpflow.kernels.SquaredExponential(lengthscales=0.3, variance=0.1),
        output_dim=X.shape[1],
        share_hyperparams=True)
    inducing_variable1 = construct_basic_inducing_variables(num_inducing, X.shape[1], X.shape[1], share_variables=True, z_init=Z.copy())
    gp_layer1 = gpflux.layers.GPLayer(kernel1, inducing_variable1, num_data=X.shape[0])

    # Layer 2
    kernel2 =  construct_basic_kernel(
        gpflow.kernels.SquaredExponential(lengthscales=0.3, variance=0.1),
        output_dim=1,
        share_hyperparams=True)
    inducing_variable2 =  construct_basic_inducing_variables(num_inducing, X.shape[1], X.shape[1], share_variables=True, z_init=Z.copy())
    gp_layer2 = gpflux.layers.GPLayer(
        kernel2,
        inducing_variable2,
        num_data=X.shape[0],
        mean_function=gpflow.mean_functions.Zero(),
    )

    # Initialise likelihood and build model
    likelihood_layer = gpflux.layers.LikelihoodLayer(gpflow.likelihoods.Gaussian(0.001))
    two_layer_dgp = gpflux.models.DeepGP([gp_layer1, gp_layer2], likelihood_layer)

    # Compile and fit
    model = two_layer_dgp.as_training_model()
    model.compile(tf.optimizers.Adam(0.01))
    history = model.fit({"inputs": X, "targets": Y}, epochs=10000, verbose=1)
    
    prediction_model = two_layer_dgp.as_prediction_model()
    train_out = prediction_model(X).f_mean.numpy()
    print(f'Train MSE {((train_out - Y)**2).mean()}')
    test_out = prediction_model(test_x).f_mean.numpy()
    print(f'Test MSE {((test_out - test_y)**2).mean()}')