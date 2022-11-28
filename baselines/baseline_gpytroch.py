import gpytorch
import torch
import os, sys
import torch.nn.functional as F
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.RBFKernel()
        # self.base_kernel = gpytorch.kernels.RBFKernel(lengthscale_constraint = gpytorch.constraints.GreaterThan(1e-6))
        # self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        

    def forward(self, x):
        '''
            Returns a PRIOR mean and a PRIOR covariance. Not the predictive one
        '''
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

class Trainer():
    def __init__(self, model,likelihood, mll, optim, train_data, test_data, device):
        self.model = model
        self.likelihood = likelihood
        self.mll = mll
        self.optimizer = optim
        self.train_data = train_data
        self.test_data = test_data
        self.device = device
        pass

    def fit(self, n_iter, verbose):
        x_train, y_train = self.train_data
        self.model.to(self.device)
        for i in range(n_iter):
            self.model.train()
            self.model.likelihood.train()
            def closure():
                self.optimizer.zero_grad()
                # Output from model
                output = model(x_train)
                # Calc loss and backprop gradients
                loss = -self.mll(output, y_train.squeeze())
                loss.backward()
                return loss

            self.optimizer.step(closure)
            loss = closure()
            if verbose:
                print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                    i + 1, n_iter, loss.item(),
                    self.model.covar_module.lengthscale,
                    self.model.likelihood.noise.item()
                ))
        test_mse = self.test()
        return test_mse


    def test(self):
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            #train loss
            x_train, y_train = self.train_data
            y_preds = []
            train_mse = 0
            self.model.eval()
            self.model.likelihood.eval()
            y_pred = self.model.likelihood(self.model(x_train))
            train_mse = F.mse_loss(y_pred.mean, y_train)
            print('train_mse', train_mse)
            
            x_test, y_test = self.test_data
            y_preds = []
            test_mse = 0
            self.model.eval()
            self.likelihood.eval()
            y_pred = self.likelihood(self.model(x_test))
            y_preds.append(y_pred.mean.cpu().detach())
            test_mse = F.mse_loss(y_pred.mean, y_test)
        # plt.plot(x_test.cpu(), y_pred)
        return test_mse

def partition_data(train_dataset, test_dataset, train_part, test_part):
    x_train = train_dataset.tensors[0][:int(train_part*len(train_dataset.tensors[1]))]
    y_train = train_dataset.tensors[1].squeeze()[:int(train_part*len(train_dataset.tensors[1]))]
    x_test = test_dataset.tensors[0][:int(test_part*len(test_dataset.tensors[0]))]
    y_test = test_dataset.tensors[1].squeeze()[:int(test_part*len(test_dataset.tensors[0]))]
    return x_train,y_train, x_test, y_test

if __name__ == '__main__':
    device='cpu'
    # regr_datasets = ['yacht', 'boston', 'concrete', 'kin8nm', 'powerplant', 'protein']
    regr_datasets = ['yacht']
    for dataset  in regr_datasets:
        train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=True)
        x_train, y_train, x_test, y_test = partition_data(train_dataset, test_dataset, train_part=1, test_part=1)
        x_train, y_train, x_test, y_test = x_train.to(device), y_train.to(device), x_test.to(device), y_test.to(device)
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.Tensor([0.09])
        model = GPModel(x_train,y_train, likelihood)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
        optim = torch.optim.LBFGS([
            {'params': model.covar_module.parameters()},  # Includes GaussianLikelihood parameters
            ], lr=0.01)
        trainer = Trainer(model, likelihood, mll, optim, (x_train, y_train), (x_test, y_test), device)
        test_mse = trainer.fit(n_iter=10, verbose=True)
        print("dataset {}, test mse {}".format(dataset, test_mse))
        
        