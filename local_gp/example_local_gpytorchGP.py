import pickle
import gpytorch
import torch
import sys
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as  plt 

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import synthetic_regression_problem
from utils import get_hs, n_neighbours_to_h, get_effective_datapoints
from kernels import epanechnikov_windowing_func, hilbert_kernel


class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        # train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_y)
         
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.RBFKernel(lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-6))
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
def construct_localGPRs(Xtr,ytr, central_points, noise, h, windowing_func):
    models = []
    for central_point_idx in range(central_points.shape[0]):
        central_point = central_points[central_point_idx]
        w = windowing_func((Xtr - central_point)/h)
        Xtr_nonzero, ytr_nonzero, w_nonzero = get_effective_datapoints(Xtr, ytr, w)
        if (w_nonzero > 0).sum() < 3:
            raise ValueError('h is too small')
        if (w_nonzero > 0).sum() > 0.5 * len(Xtr):
            raise ValueError('h is too big')

        invw_nonzero = torch.Tensor(1/w_nonzero)
        likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=invw_nonzero * noise)
        model = GPModel(Xtr_nonzero, ytr_nonzero, likelihood)
        models.append(model)
    return models


    

def train_GPR(model, lr, training_iter, verbose=False):
    model.train()
    model.likelihood.train()
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
    optimizer = torch.optim.LBFGS([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=lr)
    
    Xtr = model.train_inputs[0]
    ytr = model.train_targets
    
    def closure():
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(Xtr)
        # Calc loss and backprop gradients
        loss = -mll(output, ytr)
        loss.backward()
        return loss
    
    if verbose:
        loss = -mll(model(Xtr), ytr)
        print('initial mll {} kernel lengthscale {}'.format(loss, model.covar_module.lengthscale))
        
    for i in range(training_iter):
        optimizer.step(closure)
        loss = closure()
    if verbose:
        print('Optimized mll {},  Optimized kernel lengthscale {}'.format(loss, model.covar_module.lengthscale))





if __name__ == '__main__':
    n_points = 100
    Xtr, ytr, true_function = synthetic_regression_problem(train_len=n_points, noise_level=0.05)
    Xtr, ytr = torch.Tensor(Xtr), torch.Tensor(ytr)
    x_val = torch.Tensor(np.linspace(0,1,1000)).view(-1,1)
    y_val = true_function(x_val)
    
    noise = 0.04
    windowing_func = epanechnikov_windowing_func

    
    best_val_loss = np.inf
    best_models = []
    history = {'h':[], 'val_mse': [], 'predictions':[]}
    for h in np.logspace(-5, 0, 50):
    # for h in [0.005]:
        val_mse_loss = 0
        try:
            models = construct_localGPRs(Xtr,ytr, x_val, noise, h, windowing_func)
        except ValueError as inst:
            if inst.args[0] == 'h is too small':
                continue
            elif inst.args[0] == 'h is too big':
                break
            else:
                raise inst
        predictions = {'mean':[], 'lower': [], 'upper':[]}    
        for central_point_idx in tqdm(range(x_val.shape[0]), total=len(x_val)):
            model = models[central_point_idx]
            central_point = x_val[central_point_idx]
            train_GPR(model, lr=0.01, training_iter=10, verbose=False)
            model.eval()
            model.likelihood.eval()
            y_pred = model.likelihood(model(central_point), noise=torch.Tensor([noise]))
            predictions['mean'].append(y_pred.mean.item())
            lower, upper = y_pred.confidence_region()
            lower, upper = lower.detach().numpy().item(), upper.detach().numpy().item()
            predictions['lower'].append(lower)
            predictions['upper'].append(upper)
            val_mse_loss += (y_pred.mean - y_val[central_point_idx])**2
        val_mse_loss = val_mse_loss / len(x_val)
        print('h: {}, val mse {}'.format(h, val_mse_loss)) 
        if val_mse_loss < best_val_loss:
            best_val_loss = val_mse_loss
            best_models = models
            best_predictions = predictions
             
        history['h'].append(h)
        history['val_mse'].append(val_mse_loss)
        history['predictions'].append(predictions)  
    
    history['train_data'] = (Xtr.numpy(), ytr.numpy())
    with open(os.path.join('local_gp', 'lgp_toy{}.pkl'.format(n_points)), 'wb') as f:
        pickle.dump(history, f)

    plt.plot(x_val, y_val, label='True function')
    plt.plot(Xtr,ytr, 'kx', 'Training points')
    plt.plot(x_val, best_predictions['mean'], label='Predictions')
    plt.fill_between(x_val.reshape(-1), best_predictions['lower'], 
                     best_predictions['upper'])
    plt.legend()
    plt.show()
        