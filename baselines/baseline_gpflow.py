import gpflow
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from gpflow.utilities import print_summary

import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset, synthetic_regression_problem


def synth_exp():
    X, Y, func =  synthetic_regression_problem(train_len=100)
    X,Y = X.reshape((-1,1)), Y.reshape((-1,1))
    k = gpflow.kernels.Matern52()
    m = gpflow.models.GPR(data=(X, Y), kernel=k, mean_function=None)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=10))
    print_summary(m)
    xx = np.linspace(-0.1, 1.1, 100).reshape(100, 1)  # test points must be of shape (N, D)

    ## predict mean and variance of latent GP at test points
    mean, var = m.predict_f(xx)

    ## generate 10 samples from posterior
    tf.random.set_seed(1)  # for reproducibility
    samples = m.predict_f_samples(xx, 10)  # shape (10, 100, 1)

    ## plot
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, "kx", mew=2)
    plt.plot(xx, mean, "C0", lw=2)
    plt.fill_between(
        xx[:, 0],
        mean[:, 0] - 1.96 * np.sqrt(var[:, 0]),
        mean[:, 0] + 1.96 * np.sqrt(var[:, 0]),
        color="C0",
        alpha=0.2,
    )

    plt.plot(xx, samples[:, :, 0].numpy().T, "C0", linewidth=0.5)
    _ = plt.xlim(-0.1, 1.1)
    plt.show()

def uci_exp():
    dataset='yacht'
    train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=True)
    # Xtr, ytr = np.array(train_dataset.tensors[0], dtype=np.float64),np.array(train_dataset.tensors[1], dtype=np.float64)
    Xtr, ytr = train_dataset.tensors[0], train_dataset.tensors[1]
    Xtst, ytst = test_dataset.tensors[0], test_dataset.tensors[1]
    k = gpflow.kernels.RBF()
    m = gpflow.models.GPR(data=(Xtr, ytr), kernel=k, mean_function=None)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(m.training_loss, m.trainable_variables, options=dict(maxiter=1000))
    print_summary(m)
    mean, var = m.predict_f(Xtr.numpy().astype(np.float64))
    mean = mean.numpy()
    var = var.numpy()
    print((np.linalg.norm(mean - ytr.numpy())**2).mean())
    
    
    mean, var = m.predict_f(Xtst.numpy().astype(np.float64))
    mean = mean.numpy()
    var = var.numpy()
    print((np.linalg.norm(mean - ytst.numpy())**2).mean())
    pass
    
    
if __name__ == '__main__':
    uci_exp()
