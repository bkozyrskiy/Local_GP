import os, sys
from sklearn.kernel_ridge import KernelRidge
import sklearn
from sklearn.model_selection import KFold
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
from tqdm import tqdm
import gpytorch
import torch

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset
from utils import get_hs, n_neighbours_to_h, get_effective_datapoints
from kernels import epanechnikov_windowing_func, hilbert_kernel


    
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, lengthscale):
        train_x, train_y = torch.Tensor(train_x), torch.Tensor(train_y)
         
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()
        
        # self.base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        # RBF_kernel = gpytorch.kernels.RBFKernel(lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-6))
        # RBF_kernel.lengthscale = torch.Tensor([lengthscale])
        # self.base_kernel = RBF_kernel
        # self.covar_module = gpytorch.kernels.ScaleKernel(RectungularLocalizedKernel(h, x0, self.base_kernel, local_kernel_func))
        self.covar_module = gpytorch.kernels.RBFKernel()
        self.covar_module.lengthscale =  torch.Tensor([lengthscale])

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def construct_GPR(Xtr,ytr, central_point, lengthscale, noise, n_points, windowing_func):
    central_point = central_point.reshape(1,-1)
    h = n_neighbours_to_h(central_point, Xtr, n_points=n_points)
    w = windowing_func((Xtr - central_point)/h)
    Xtr_nonzero, ytr_nonzero, w_nonzero = get_effective_datapoints(Xtr, ytr, w)
    invw_nonzero = torch.Tensor(1/w_nonzero)
    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(noise=invw_nonzero * noise)
    model = GPModel(Xtr_nonzero, ytr_nonzero, likelihood, lengthscale)
    return model, likelihood
    


def test(Xtr, ytr, Xtst, ytst, params, windowing_func):
    ytr = ytr.reshape(-1)
    
    n_points = params["n_points"]
    lengthscale = params['lengthscale']
    noise = params['noise']
    
    center_mse = 0
    for idx_center in range(ytst.shape[0]):
        model, likelihood = construct_GPR(Xtr,ytr, Xtst[idx_center], lengthscale, noise, n_points, windowing_func)
        model.eval()
        model.likelihood.eval()
        # y_center_pred = model.predict(Xtst[idx_center].reshape(1,-1))
        central_point = torch.Tensor(Xtst[idx_center]).reshape(1,-1)
        y_center_pred = likelihood(model(central_point), noise=torch.Tensor([noise])).mean.cpu().detach().numpy()
        center_mse += (y_center_pred - ytst[idx_center])**2
    center_mse = center_mse/len(ytst)
    return center_mse

# def cv_test(X,y, params, windowing_func, k_folds=4):
#     kf = KFold(n_splits=k_folds)
    

if __name__ == "__main__":
    # datasets = ['yacht', 'boston','concrete', 'kin8nm', 'powerplant', 'protein']
    datasets = ['concrete']
    for dataset in datasets:
        print(dataset)
        train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=False)
        Xtr, ytr = np.array(train_dataset.tensors[0]), np.array(train_dataset.tensors[1])
        Xtst, ytst = np.array(test_dataset.tensors[0]), np.array(test_dataset.tensors[1])
        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xtst = scaler.fit_transform(Xtst)
        
        # Xtr, ytr = torch.Tensor([Xtr]), torch.Tensor([ytr]) 
        # Xtst, ytst = torch.Tensor([Xtst]), torch.Tensor([ytst]) 
        
        best_parameters = {"noise": 0.0112,
                        "lengthscale": 0.456,
                        "n_points": 100}
        
        
        center_mse = test(Xtr, ytr, Xtst, ytst, best_parameters, windowing_func=hilbert_kernel)
        print('test mse {}'.format(center_mse))
        print(best_parameters)