import gpytorch
import torch
import torch.nn.functional as F
from datasets import synthetic_regression_problem
import matplotlib.pyplot as plt
import numpy as np

    
def windowing_func(x):
    return torch.abs(x) < 1
# class LocalKernel(gpytorch.kernels.Kernel):
#     def __init__(self, h, x0, kernel_func):
#         super().__init__()
#         self.x0 = x0 
#         self.h = h
#         self.kernel_func = kernel_func
#     def forward(self, x1, x2, diag=False, last_dim_is_batch=False):
#         # left = torch.sqrt(self.kernel_func(x1,self.x0))
#         # right = torch.sqrt(self.kernel_func(self.x0,x2))
#         return torch.outer(((x1-self.x0)**2<self.threshold).type(torch.float).squeeze(),((x2-self.x0)**2<self.threshold).type(torch.float).squeeze())

class LocalizedKernel(gpytorch.kernels.Kernel):
    def __init__(self, h, x0, base_kernel):
        super().__init__()
        self.x0 = x0 
        # self.h = torch.Tensor([h])
        self.h = h
        self.base_kernel = base_kernel
        self.local_kernel_func = lambda x: torch.abs(x) < self.h
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        left = torch.sqrt(1/np.exp(self.h) * self.local_kernel_func((x1 - self.x0) / np.exp(self.h)))
        right = torch.sqrt(1/np.exp(self.h) * self.local_kernel_func((self.x0 - x2) / np.exp(self.h)))
        return left.view(-1,1) * self.base_kernel(x1, x2, diag, last_dim_is_batch, **params).evaluate() * right.view(1,-1)
        
    
class LocalGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, h, x0):
        super(LocalGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = LocalizedKernel(h,x0,self.base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def test(models,x_test):
    y_pred = []
    for idx, (model,local_idx, mll, optimizer) in enumerate(models):
        model.eval()
        y_pred.append(model(x_test[idx]).mean.cpu().detach())
    plt.plot(x_test.cpu(),y_pred)
    pass
        

# class Trainer():
#     def __init__(self,):
#         self.test_batch_size = test_batch_size
    
#     def fit(self,x_test):
        
        
        

if __name__ == '__main__':
    device = 'cuda:1'
    x_train, y_train, true_function = synthetic_regression_problem(train_len=800)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    x_test = torch.Tensor(np.linspace(0,1,1000)).view(-1,1).to(device)
    #x_test = torch.Tensor([0.5]).view(-1,1).to(device)
    x_train, y_train = x_train.to(device), y_train.to(device)
    models = []
    h = -5
    learnable_parameters = []
    for x in x_test:
        local_idx = windowing_func((x_train - x) / np.exp(h)).nonzero().squeeze()
        model = LocalGPModel(x_train[local_idx], y_train[local_idx], likelihood, h=h, x0=x)
        model.train()
        likelihood.train()
        # optimizer = torch.optim.Adam([
        #     {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        #     ], lr=0.1)
        learnable_parameters += list(model.parameters())
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        model.to(device), likelihood.to(device), mll.to(device)
        models.append((model,local_idx, mll))
    optimizer = torch.optim.Adam(learnable_parameters, lr=0.1) 
    training_iter = 50
    for i in range(training_iter):
        loss = 0 
        for model, local_idx, mll in models: 
            # Zero gradients from previous iteration
            optimizer.zero_grad()
            # Output from model
            output = model(x_train[local_idx])
            # Calc loss and backprop gradients
            loss += -mll(output, y_train[local_idx])
        loss.backward()
        print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
            i + 1, training_iter, loss.item(),
            model.covar_module.base_kernel.base_kernel.lengthscale,
            model.likelihood.noise.item()
        ))
        optimizer.step()
    test(models, x_test)
    pass

# if __name__ == '__main__':
#     base_kernel = gpytorch.kernels.RBFKernel()
#     localized_kernel = LocalizedKernel(1e-3, 0.5, base_kernel)
#     pass
          
