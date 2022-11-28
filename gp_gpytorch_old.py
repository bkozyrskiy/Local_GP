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
from kernels import rectangular_windowing_func, epanechnikov_windowing_func, RectungularLocalizedKernel, EpanechnikovLocalizedKernel, rectangular_windowing_func_old


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




class LocalGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, h, x0, local_kernel_func):
        super(LocalGPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()
        RBF_kernel = gpytorch.kernels.RBFKernel(lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-6))
        self.base_kernel = RBF_kernel
        # self.covar_module =\
            # gpytorch.kernels.ScaleKernel(RectungularLocalizedKernel(h, x0, self.base_kernel, local_kernel_func))
            
        # self.covar_module =\
        #     gpytorch.kernels.ScaleKernel(EpanechnikovLocalizedKernel(h, x0, self.base_kernel,
        #                                                              local_kernel_func,optimizable_scale=False))
        self.covar_module = RectungularLocalizedKernel(h, x0, self.base_kernel, local_kernel_func)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def get_models_list(x_train, y_train, x_test, val_size, h, lr, alpha, windowing_func):
    models = []
    ns_train_points = []
    for x in x_test:
        local_idx = windowing_func((x_train - x) / h).nonzero().squeeze()
        if (local_idx.dim() == 0) or (len(local_idx) <= 2):
            print("Too few within this distance")
            raise ValueError
        if val_size is not None:
            train_local_idx, val_local_idx = train_test_split(local_idx, test_size=val_size) 
        else:
            train_local_idx, val_local_idx = local_idx, None
        ns_train_points.append(len(local_idx))    
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.Tensor([alpha])
        model = LocalGPModel(x_train[train_local_idx], y_train[train_local_idx], likelihood, h=h, x0=x, 
                             local_kernel_func=windowing_func)
        model.train()
        optimizer = torch.optim.LBFGS([
            {'params': model.covar_module.parameters()},  # Includes GaussianLikelihood parameters
        ], lr=lr)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        # model.to(device), likelihood.to(device), mll.to(device)
        models.append((model, (train_local_idx, val_local_idx), mll, optimizer))
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
        nll = -mll(y, pred)
        return mse, nll
    
        
    def fit(self,n_iter, verbose):
        x_train, y_train = self.train_data
        x_train, y_train = x_train.to(self.device), y_train.to(self.device)
        for model, (train_local_idx, val_local_idx), mll, optimizer in self.models_list:
            model.to(self.device), mll.to(self.device)
            for i in range(n_iter):
                model.train()
                model.likelihood.train()
                # Zero gradients from previous iteration
                def closure():
                    optimizer.zero_grad()
                    # Output from model
                    output = model(x_train[train_local_idx])
                    # Calc loss and backprop gradients
                    loss = -mll(output, y_train[train_local_idx].squeeze())
                    loss.backward()
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
            if val_local_idx is not None:
                val_mse, val_mll = self.evaluate_model(model,mll, x_train[val_local_idx], y_train[val_local_idx])
            model.to('cpu'), mll.to('cpu')
            torch.cuda.empty_cache()
        train_mse, test_mse = self.test()
        return train_mse, test_mse


    def test(self):
        x_train, y_train = self.train_data
        x_test, y_test = self.test_data
        # y_preds = []
        train_mse = 0
        test_mse = 0
        for idx, (model, (train_local_idx, val_local_idx), mll, optimizer) in enumerate(self.models_list):
            # model.to(self.device), mll.to(self.device)
            model.eval()
            model.likelihood.eval()
            pred_train = model(x_train[train_local_idx])
            # train_mse += (torch.norm(pred_train.mean - y_train[train_local_idx])**2).item()
            train_mse = F.mse_loss(pred_train.mean, y_train[train_local_idx])
            
            
            pred_test = model(x_test[idx].view(1,-1))
            test_mse += (torch.norm(pred_test.mean - y_test[idx])**2).item()
            
            
            model.to('cpu'), mll.to('cpu')
            torch.cuda.empty_cache()
        train_mse /= len(y_train)
        test_mse /= len(self.models_list)
        # plt.plot(x_test.cpu(), y_pred)
        return train_mse, test_mse

# def test_imlementation(lengthscale=0.267):
#     device = 'cuda:1'
#     dataset = 'yacht'
#     train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardize=True)
#     x_train, y_train = train_dataset.tensors[0].to(device), train_dataset.tensors[1].squeeze().to(device)
#     x_test, y_test = test_dataset.tensors[0].to(device), test_dataset.tensors[1].squeeze().to(device)
#     h = np.inf
#     print("h {}".format(h))
#     RBF_lengthscale = lengthscale
#     models = get_models_list(x_train, y_train, x_test, h, RBF_lengthscale, device)
#     y_preds = []
#     test_mll = 0
#     test_mse = 0
#     for idx, (model, local_idx, mll, optimizer) in enumerate(models):
#         model.eval()
#         y_pred = model(x_test[idx].view(1,-1))
#         y_preds.append(y_pred.mean.cpu().detach())
#         test_mll = -mll(y_pred, y_test[idx])
#         test_mse += torch.norm(y_pred.mean - y_test[idx])**2
#     test_mse /= len(y_test)
#     print(test_mse)
#     return test_mse
    
    


if __name__ == '__main__':
    # test_imlementation(lengthscale=1)
    device = 'cuda:0'
    dataset = 'concrete'
    train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=True)
    # x_train, y_train = train_dataset.tensors[0][:int(0.1*len(train_dataset.tensors[1]))], train_dataset.tensors[1].squeeze()[:int(0.1*len(train_dataset.tensors[1]))]
    # x_test, y_test = test_dataset.tensors[0][:int(0.1*len(test_dataset.tensors[0]))], test_dataset.tensors[1].squeeze()[:int(0.1*len(test_dataset.tensors[0]))]
    x_train, y_train = train_dataset.tensors[0], train_dataset.tensors[1].squeeze()
    x_test, y_test = test_dataset.tensors[0], test_dataset.tensors[1].squeeze()
    for logh in np.arange(-5,5,0.5):
        print("logh {}".format(logh))
        h = np.exp(logh)
        try:
            models, ns_train_points = get_models_list(x_train, y_train, x_test, 
                                                      None, h, lr=0.1, alpha=0.2,
                                                      windowing_func=rectangular_windowing_func)
        except ValueError:
            continue
        train = Trainer(models, (x_train, y_train), (x_test, y_test), device)
        train_mse, test_mse = train.fit(n_iter=4, verbose=False)
        print("h {},train_mse {}, test_mse {}".format(h, train_mse, test_mse))
        pass

    


