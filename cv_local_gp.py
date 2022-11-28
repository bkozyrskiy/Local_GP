import gpytorch
import numpy as np
import gpytorch
from sklearn.model_selection import train_test_split, KFold
import torch.nn.functional as F
import torch
import copy
from datasets import general_torch_dataset
from kernels import rectangular_windowing_func, rectangular_windowing_func_old, RectungularLocalizedKernel
from tqdm import tqdm



class LocalGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, h, x0, local_kernel_func):
        super(LocalGPModel, self).__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()
        # self.base_kernel = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        RBF_kernel = gpytorch.kernels.RBFKernel(lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-6))
        self.base_kernel = RBF_kernel
        # self.covar_module = gpytorch.kernels.ScaleKernel(RectungularLocalizedKernel(h, x0, self.base_kernel, local_kernel_func))
        self.covar_module = RectungularLocalizedKernel(h, x0, self.base_kernel, local_kernel_func)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
        
        

class LocalModelTrainer():
    def __init__(self, central_point, train_data,  windowing_func, device) -> None:
        self.central_point = central_point
        self.train_data = train_data
        self.device = device
        self.windowing_func = windowing_func
    
    def get_model(self, local_idx, h, lr, alpha):
        '''
            alpha: regularization paprameter, equivalent to \sigma^2
        '''
        x_train, y_train = self.train_data
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.Tensor([alpha])
        model = LocalGPModel(x_train[local_idx], y_train[local_idx], likelihood, h=h, x0=self.central_point, 
                             local_kernel_func=self.windowing_func)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
        optimizer = torch.optim.LBFGS([
            {'params': model.covar_module.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=lr)
        return model, mll, optimizer
        
    def get_local_train_idx(self, h):
        x_train, y_train = self.train_data
        local_idx = self.windowing_func((self.central_point - x_train) / h).nonzero().squeeze()
        if (local_idx.dim() == 0) or (len(local_idx) <= 10):
            # print("Too few within this distance")
            raise ValueError("Too few within this distance")
        elif len(local_idx) == len(x_train):
            raise ValueError("More than half of the points within this distance")
        return local_idx
    
    def evaluate_model(self, model, mll_func, x, y):
        model.eval()
        model.likelihood.eval()
        pred = model(x)
        mse = F.mse_loss(pred.mean, y, reduction='mean')
        nmll = -mll_func(pred, y)
        return mse.item(), nmll.item()
    
    def h_validation(self,logh_range, n_iter, val_size, lr, alpha):
        self.best_h = None
        min_nmll = np.inf
        min_mse = np.inf
        for logh in logh_range:
            h = np.exp(logh)
            try:
                local_idx = self.get_local_train_idx(h)
            except ValueError:
                continue
            # train_local_idx, val_local_idx = train_test_split(local_idx, test_size=val_size, random_state=0)
            # model, mll, optimizer = self.get_model(train_local_idx, h)
            # model, fold_train_loss, val_mse, val_nll = self.optimize(n_iter=4, model=model, mll=mll, optimizer=optimizer,
            #                                                         train_local_idx=train_local_idx, val_local_idx=val_local_idx)
            kf = KFold(n_splits=5)
            folds_val_nmll = []
            folds_val_mse = []
            for train_idx_idx, val_idx_idx in kf.split(local_idx):
                train_local_idx = local_idx[train_idx_idx]
                val_local_idx = local_idx[val_idx_idx]
                model, mll, optimizer = self.get_model(train_local_idx, h, lr, alpha)
                model.to(self.device), mll.to(self.device)
                fold_train_nmll, fold_val_mse, fold_val_nmll = self.optimize(n_iter=n_iter, model=model, mll=mll, optimizer=optimizer,
                                                                            val_local_idx=val_local_idx)
                folds_val_nmll.append(fold_val_nmll)
                folds_val_mse.append(fold_val_mse)
                
            val_nmll = np.mean(folds_val_nmll)
            val_mse = np.mean(folds_val_mse)
            if val_nmll < min_nmll:
                min_nmll = val_nmll
                min_mse = val_mse
                self.best_h = h
                effective_n_train_points = len(train_local_idx)
                self.best_model = copy.deepcopy(model)
        return min_mse, min_nmll, self.best_h, effective_n_train_points, 
            
    def optimize(self, n_iter, model, mll, optimizer, val_local_idx):
        x_train, y_train = self.train_data
        for i in range(n_iter):
            model.train()
            model.likelihood.train()
            def closure():
                optimizer.zero_grad()
                # Output from model
                output = model(model.train_inputs[0])
                # Calc loss and backprop gradients
                loss = -mll(output, model.train_targets[0].squeeze())
                loss.backward()
                return loss
            
            optimizer.step(closure)
            tmp_train_nmll = closure()
            tmp_train_nmll = tmp_train_nmll.item()
        if val_local_idx is not None:
            val_mse, val_nmll = self.evaluate_model(model, mll, x_train[val_local_idx], y_train[val_local_idx])
            
        else:
            val_mse, val_nmll = None, None
                
        return tmp_train_nmll, val_mse, val_nmll
            
         
                
    def test_best_copy(self,central_point_label):
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.best_model.likelihood, self.best_model)
        test_mse, test_nll = self.evaluate_model(self.best_model, mll, self.central_point.view(1,-1), central_point_label.view(1))
        return test_mse, test_nll
    
    
if __name__ == '__main__':
    device = 'cuda:1'
    # dataset = 'concrete'
    for dataset in ['protein']:
        train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=True)
        train_data_part = 1
        test_data_part = 0.1
        x_train = train_dataset.tensors[0][:int(train_data_part * len(train_dataset.tensors[0]))].to(device)
        y_train = train_dataset.tensors[1].squeeze()[:int(train_data_part * len(train_dataset.tensors[1]))].to(device)
        x_test = test_dataset.tensors[0][:int(test_data_part * len(test_dataset.tensors[0]))].to(device)
        y_test = test_dataset.tensors[1][:int(test_data_part * len(test_dataset.tensors[1]))].squeeze().to(device)
        test_mses, test_nmlls = [], []
        val_mse, val_nll = 0, 0 
        hs_list, ns_train_points = [],[]
        for test_idx, x in tqdm(enumerate(x_test), total=len(x_test)):
            lmtrainer = LocalModelTrainer(x, (x_train,y_train), rectangular_windowing_func, device)
            tmp_val_mse, tmp_val_nll, best_h, effective_n_train_points =\
                lmtrainer.h_validation(logh_range=np.arange(-5,5,0.5), n_iter=4, val_size=0.2, lr=0.001, alpha=0.2)
            val_mse += tmp_val_mse
            val_nll += tmp_val_nll
            hs_list.append(best_h)
            ns_train_points.append(effective_n_train_points)
            tmp_test_mse, tmp_test_nmll = lmtrainer.test_best_copy(y_test[test_idx])
            test_mses.append(tmp_test_mse)
            test_nmlls.append(tmp_test_nmll)
            del lmtrainer
            torch.cuda.empty_cache()
        val_mse /= len(x_test)
        print(dataset)
        print('h list', hs_list)
        print('Number of training points', ns_train_points)
        print('Val MSE {}, NLL {}'.format(val_mse, val_nll))
        print('Test MSE {}, NLL {}'.format(np.mean(test_mses), np.sum(test_nmlls)))
        pass
    