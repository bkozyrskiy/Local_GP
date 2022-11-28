TORCHGP_PATH = '/home/bogdan/ecom/torchgp'

import numpy  as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset, synthetic_regression_problem

sys.path.append(TORCHGP_PATH)
from torchgp.gp import GPR, FeatureGPR
from torchgp.kernels import RBF
from torchgp.optim import LBFGS, Adagrad
from torchgp.features import RFF


def synth_exp():
    X, y, func =  synthetic_regression_problem(train_len=100, noise_level=0.01)

    leng=0.28
    sn2=0.1

    rff = RFF(X.shape[1], 1000, leng=leng, ARD=False)
    gpr = FeatureGPR(X, y, phi_func=rff, weight_var=1, sn2=sn2)
    steps = 10000
    threshold = 1e-1
    #opt = Adagrad(steps=steps, threshold=threshold, lr=0.1)
    opt = LBFGS(steps=steps, threshold=threshold, lr=0.2)
    opt.optimise(gpr, plot=True)


    Xtest = np.linspace(-0.1, 1.1 ,1000).reshape(-1,1)
    _, _, mu, s2 = gpr.predict(Xtest)

    lb = mu - 2 * np.sqrt(s2)
    ub = mu + 2 * np.sqrt(s2)


    plt.figure()
    plt.plot(X, y, 'o')
    plt.plot(Xtest, mu, 'b')
    plt.fill_between(Xtest.flatten(), ub, lb, facecolor='0.75')
    plt.show()

if __name__ == '__main__':
    dataset = 'yacht'
    train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset_name=dataset, standardise=True)
    x_train, y_train = train_dataset.tensors[0], train_dataset.tensors[1]
    x_test, y_test = test_dataset.tensors[0], test_dataset.tensors[1]
    # leng=1.187
    # sn2=0.09
    
    leng=0.5
    sn2=0.1
    rff = RFF(x_train.shape[1], 1000, leng=leng, ARD=False)
    gpr = FeatureGPR(x_train, y_train, phi_func=rff, weight_var=1, sn2=sn2)
    gpr.logsn2.set_optimise(False)
    gpr.logwvar.set_optimise(False)
    gpr.forward()
    # steps = 10000
    # threshold = 1e-1
    # opt = Adagrad(steps=steps, threshold=threshold, lr=0.1)
    # opt = LBFGS(steps=steps, threshold=threshold, lr=0.01)
    # opt.optimise(gpr, plot=True)
    
    _, _, mu_train, s2_train = gpr.predict(x_train)
    _, _, mu_test, s2_test = gpr.predict(x_test)
    print('MSE train', F.mse_loss(mu_train, y_train.reshape(-1)))
    print('MSE test', F.mse_loss(mu_test, y_test.reshape(-1)))
    
