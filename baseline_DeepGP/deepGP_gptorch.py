
import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO, AddedLossTerm
from gpytorch.likelihoods import GaussianLikelihood, FixedNoiseGaussianLikelihood
from sklearn.preprocessing import MinMaxScaler

from torch.utils.data import DataLoader, TensorDataset

from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL


import gpytorch
import torch
import os, sys
import torch.nn.functional as F
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset
from utils import standardise_labels

class ToyDeepGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, mean_type='constant'):
        if output_dims is None:
            inducing_points = torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(ToyDeepGPHiddenLayer, self).__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
        else:
            self.mean_module = LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )
        # self.covar_module = ScaleKernel(
        #     RBFKernel(batch_shape=batch_shape, ard_num_dims=None),
        #     batch_shape=batch_shape, ard_num_dims=None
        # )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))
    
    
class DeepGP(DeepGP):
    def __init__(self, train_x_shape, num_output_dims):
        hidden_layer = ToyDeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_output_dims,
            mean_type='linear',
        )

        last_layer = ToyDeepGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=None,
            mean_type='constant',
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer
        self.likelihood = GaussianLikelihood()
        # self.likelihood = FixedNoiseGaussianLikelihood(noise=1e-2 * torch.ones(train_x_shape[0]), learn_additional_noise=False)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(model.likelihood.log_marginal(y_batch, model(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)
    
if __name__ == '__main__':
    dataset = 'yacht'
    train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=False)
    train_x, train_y = train_dataset.tensors[0], train_dataset.tensors[1].squeeze()
    test_x, test_y = test_dataset.tensors[0], test_dataset.tensors[1].squeeze()
    
    scaler = MinMaxScaler()
    scaler.fit(train_x)
    train_x = torch.Tensor(scaler.transform(train_x))
    test_x = torch.Tensor(scaler.fit_transform(test_x))
    
    train_y, mean, std = standardise_labels(train_y)
    
    train_loader = DataLoader(TensorDataset(train_x,train_y), batch_size=1024, shuffle=True)
    
    model = DeepGP(train_x.shape, output_dim)
    # if torch.cuda.is_available():
    #     model = model.cuda()
        
    optimizer = torch.optim.Adam([{'params': model.parameters()},], lr=0.1)
    mll = DeepApproximateMLL(VariationalELBO(model.likelihood, model, train_x.shape[-2]))
    num_epochs = 500
    num_samples = 10
    # epochs_iter = tqdm.notebook.tqdm(range(num_epochs), desc="Epoch")
    
    for i in tqdm.tqdm(range(num_epochs), desc="Epoch"):
        # Within each iteration, we will go over each minibatch of data
        minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
        for x_batch, y_batch in minibatch_iter:
            with gpytorch.settings.num_likelihood_samples(num_samples):
                optimizer.zero_grad()
                output = model(x_batch)
                loss = -mll(output, y_batch.reshape(-1))
                loss.backward()
                optimizer.step()
                print(f'neg mll {loss}')
                minibatch_iter.set_postfix(loss=loss.item())

    
    
    test_loader = DataLoader(TensorDataset(test_x, test_y), batch_size=1024)
    model.eval()
    train_predictive_means, train_predictive_variances, train_lls = model.predict(train_loader)
    test_predictive_means, test_predictive_variances, test_lls = model.predict(test_loader)
   
    train_rmse = (torch.mean(train_predictive_means.mean(0)* std - train_dataset.tensors[1] * std)**2).sqrt()
    print(f"train RMSE: {train_rmse.item()}, NLL: {-train_lls.mean().item()}")
    
    test_rmse = torch.mean(torch.pow(test_predictive_means.mean(0)*std + mean - test_dataset.tensors[1], 2)).sqrt()
    print(f"test RMSE: {test_rmse.item()}, NLL: {-test_lls.mean().item()}")