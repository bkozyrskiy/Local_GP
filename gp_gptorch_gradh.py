from warnings import catch_warnings
import gpytorch
from gpytorch.settings import verbose_linalg
import torch
from torch._C import Value
import torch.nn.functional as F
from datasets import synthetic_regression_problem, general_torch_dataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from kernels import epanechnikov_windowing_func, EpanechnikovLocalizedKernel


class LocalGPModel(gpytorch.models.ExactGP):
    def __init__(self, full_train_x, full_train_y, likelihood, h, x0, local_kernel_func):
        self.local_kernel_func = local_kernel_func
        self.x0 = x0
        train_x, train_y = self.get_train_subset(full_train_x, full_train_y, h)
        super(LocalGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module = gpytorch.means.ZeroMean()
        self.base_kernel = gpytorch.kernels.RBFKernel(lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-6))
            
        self.covar_module =\
            gpytorch.kernels.ScaleKernel(EpanechnikovLocalizedKernel(h, x0, self.base_kernel,
                                                                     local_kernel_func,optimizable_scale=True))
    def get_train_subset(self, full_train_x, full_train_y, h):
        '''
        Finds training points, that belongs to neighbourhood of the central point
        '''
        local_idx = self.local_kernel_func((full_train_x - self.x0) / h).nonzero().squeeze()
        if (local_idx.dim() == 0) or (len(local_idx) <= 2):
            print("Too few within this distance")
            raise ValueError
        self.local_idx = local_idx
        return full_train_x[local_idx], full_train_y[local_idx]
    
    def update_train_subset(self, full_train_x, full_train_y):
        '''updates train subset of the model according to a new value of h from localized kernel'''
        h = torch.exp(self.covar_module.base_kernel.logh)
        train_x, train_y = self.get_train_subset(full_train_x, full_train_y, h)
        train_x, train_y = train_x.to(self.train_inputs[0].device), train_y.to(self.train_inputs[0].device)
        self.set_train_data(inputs=train_x, targets=train_y, strict=False)
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_models_list(x_train, y_train, x_test, h, lr, windowing_func, device):
    models = []
    ns_train_points = []
    for x in x_test:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        model = LocalGPModel(x_train, y_train, likelihood, h=h, x0=x, 
                             local_kernel_func=windowing_func)
        ns_train_points.append(len(model.train_inputs[0])) 
        model.train()
        optimizer = torch.optim.LBFGS([
            {'params': model.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        model.to(device), likelihood.to(device), mll.to(device)
        models.append((model, mll, optimizer))
    print("ns effective training points", ns_train_points)
    return models, ns_train_points
    
class Trainer():
    def __init__(self, models_list, train_data, test_data, device):
        self.models_list = models_list
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        pass

    def evaluate_model(self, model, mll, x, y):
        pred = model(x)
        mse = F.mse_loss(pred,x, reduction='mean')
        n_marginal_ll = -mll(y, pred)
        return mse, n_marginal_ll
    
        
    def fit(self,n_iter, verbose):
        full_x_train, full_y_train = self.train_data
        for idx, (model, mll, optimizer) in enumerate(self.models_list):
            for i in range(n_iter):
                model.train()
                # Zero gradients from previous iteration
                def closure():
                    optimizer.zero_grad()
                    # Output from model
                    output = model(model.train_inputs[0])
                    # Calc loss and backprop gradients
                    loss = -mll(output,model.train_targets)
                    loss.backward()
                    model.update_train_subset(full_x_train, full_y_train)
                    return loss

                optimizer.step(closure)
                tmp_train_loss = closure()
                tmp_train_loss = tmp_train_loss.item()
        
                if verbose:
                    # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    #     i + 1, n_iter, loss.item(),
                    #     model.covar_module.base_kernel.base_kernel.lengthscale,
                    #     model.likelihood.noise.item()
                    # ))
                    print('Iter %d/%d - Loss: %.3f  noise: %.3f' % (
                        i + 1, n_iter, tmp_train_loss,
                        model.likelihood.noise.item()
                    ))
            # if val_local_idx is not None:
            #     mse, n_marginal_ll = self.evaluate_model(model,mll, x_train[val_local_idx], y_train[val_local_idx])
        test_mll, test_mse = self.test()
        return test_mll, test_mse


    def test(self):
        x_test, y_test = self.test_data
        y_preds = []
        test_n_marginal_ll = 0
        test_mse = 0
        for idx, (model, mll, optimizer) in enumerate(self.models_list):
            model.eval()
            y_pred = model(x_test[idx].view(1,-1))
            y_preds.append(y_pred.mean.cpu().detach())
            test_n_marginal_ll += -mll(y_pred, y_test[idx])
            test_mse += torch.norm(y_pred.mean - y_test[idx])**2 
            # test_mse += F.mse_loss(y_pred.mean,y_test[idx])
        test_mse /= len(y_test)
        # plt.plot(x_test.cpu(), y_pred)
        return test_n_marginal_ll, test_mse


if __name__ == '__main__':
    # test_imlementation(lengthscale=1)
    device = 'cuda:0'
    dataset = 'boston'
    train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardize=True)
    x_train, y_train = train_dataset.tensors[0].to(device), train_dataset.tensors[1].squeeze().to(device)
    x_test, y_test = test_dataset.tensors[0].to(device), test_dataset.tensors[1].squeeze().to(device)
    # x_test, y_test = x_test[19:20], y_test[19:20]
    # for logh in np.arange(-5,5,0.5):
    logh = 1.9
    print("logh {}".format(logh))
    h = np.exp(logh)
    # try:
    models, ns_train_points = get_models_list(x_train, y_train, x_test,
                                              h=h, lr=0.01, windowing_func=epanechnikov_windowing_func, device=device)
    # except ValueError:
    #     continue
    train = Trainer(models, (x_train, y_train), (x_test, y_test), device)
    test_n_marginal_ll, test_mse = train.fit(n_iter=4, verbose=False)
    print("h {},test_n_marginal_ll {}, test_mse {}".format(h, test_n_marginal_ll, test_mse))
    pass

    


