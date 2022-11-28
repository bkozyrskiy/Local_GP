import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import torch.nn.functional as F
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle

import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset, standardize_data, synthetic_regression_problem


def standardise_labels(y):
    mean = torch.mean(y, axis=0)
    std = torch.std(y, axis=0)
    y_normed = (y - mean)/std
    return y_normed, mean, std
    

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lenghtscale=None):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.RBFKernel()
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def synth_exp():
    n_points = 100
    train_x, train_y, func =  synthetic_regression_problem(train_len=n_points, noise_level=0.05)
    train_x = torch.Tensor(train_x)
    train_y = torch.Tensor(train_y)
    noise = 0.05
    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=torch.Tensor([noise]))
    model = ExactGPModel(train_x, train_y, likelihood)
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    training_iter = 100
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.lengthscale.item(),
            model.likelihood.noise.item()
        ))
        optimizer.step()
    
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    
    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0, 1, 1000)
        observed_pred = likelihood(model(test_x), noise=noise*torch.ones_like(test_x))
        
    with torch.no_grad():
        # Initialize plot
        test_y = func(test_x)
        f, ax = plt.subplots(1, 1, figsize=(4, 3))

        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'r', label='predictions')
        
        ax.plot(test_x.numpy(), test_y.numpy(), 'b', label='true fucntion')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        # ax.legend(['Observed Data', 'Mean', 'Confidence'])
        ax.legend()
        plt.show()
    toyGP_history = {'train_data':(train_x.numpy(), train_y.numpy())}
    toyGP_history['test_data'] = (test_x.numpy(),test_y.numpy())
    toyGP_history['predictions'] = (observed_pred.mean, observed_pred.confidence_region())
    with open(os.path.join('baselines', 'gp_toy{}.pkl'.format(n_points)), 'wb') as f:
        pickle.dump(toyGP_history, f)

def uci_exp(dataset, verbose):
    train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset_name=dataset, standardise=True)
    train_x, train_y = train_dataset.tensors[0], train_dataset.tensors[1].squeeze()
    test_x, test_y = test_dataset.tensors[0], test_dataset.tensors[1].squeeze()
    # train_y, mean, std = standardise_labels(train_y)
    mean, std = 0,1
    # initialize likelihood and model
    # likelihood = gpytorch.likelihoods.GaussianLikelihood()
    noise = 0.09
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(torch.ones_like(train_y)*noise)
    model = ExactGPModel(train_x, train_y, likelihood)
    
    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    
    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    # gamma = 0.1
    # # gamma = 10
    # print('Good lengthscale', 1/np.sqrt(gamma))
    # model.covar_module.lengthscale = torch.Tensor([1/np.sqrt(gamma)])

    
    training_iter = 200
    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        if verbose:
            print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f' % (
                i + 1, training_iter, loss.item(),
                model.covar_module.lengthscale.item(),
                # model.covar_module.lengthscale.item(),
                #model.likelihood.noise.item()
            ))
        optimizer.step()
    
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()
    
    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = likelihood(model(train_x))
        print('Train mse',F.mse_loss(observed_pred.mean*std + mean, train_y*std + mean))
        
        observed_pred = likelihood(model(test_x), noise=torch.zeros(len(test_x))*noise)
        print('Test mse', F.mse_loss(observed_pred.mean*std + mean, test_y))

if __name__ == '__main__':
    synth_exp()
    # for dataset in ['yacht', 'boston', 'concrete']:
    # for dataset in ['yacht']:
    #     print(dataset)
    #     uci_exp(dataset, verbose=True)
    

    