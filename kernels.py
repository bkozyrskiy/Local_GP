import torch.nn as nn
from typing import NoReturn
import torch
import gpytorch
import numpy as np
import math


def rectangular_windowing_func(x):
    return torch.norm(x,dim=-1) < 1

def hilbert_kernel(x):
    if type(x) == np.ndarray:
        if len(x.shape) == 2:
            return (1 / np.linalg.norm(x,axis=-1))* (np.linalg.norm(x,axis=-1) <= 1)
        if len(x.shape) == 1:
            return 1 / np.abs(x)* (np.abs(x) <= 1)
        
    elif type(x) == torch.Tensor:
        if len(x.shape) == 2:
            return (1 / torch.linalg.norm(x,axis=-1))* (torch.linalg.norm(x,axis=-1) <= 1)
        if len(x.shape) == 1:
            return 1 / torch.abs(x)* (torch.abs(x) <= 1)
    
def epanechnikov_windowing_func(x):
    
    def unit_sphere_V(d):
            return (np.pi**(d/2)/math.gamma(1 + d/2))
        
    if type(x) == np.ndarray:
        if len(x.shape) == 2:
            num = x.shape[1]+2
            denum = 2 * unit_sphere_V(x.shape[1])
            return (num/denum) * (1 - np.linalg.norm(x,axis=1)**2) * (np.linalg.norm(x,axis=1) <= 1)
        
        if len(x.shape) == 1:
            return (3/4) * (1 - np.linalg.abs(x)) * (np.linalg.abs(x) <= 1)
        
    elif type(x) == torch.Tensor:
        if len(x.shape) == 2:
            num = x.shape[1]+2
            denum = 2 * unit_sphere_V(x.shape[1])
            return (num/denum) * (1 - torch.linalg.norm(x,axis=1)**2) * (torch.linalg.norm(x,axis=1) <= 1)
        
        if len(x.shape) == 1:
            return (3/4) * (1 - torch.linalg.abs(x)) * (torch.linalg.abs(x) <= 1)
    else:
        raise ValueError('Wrong data type')
        
    
    
    
# class RectungularLocalizedKernel(gpytorch.kernels.Kernel):
#     def __init__(self, h, x0, base_kernel,local_kernel_func):
#         super().__init__()
#         self.x0 = nn.Parameter(x0,requires_grad=False)
#         # self.h = torch.Tensor([h])
#         self.h = h
#         self.base_kernel = base_kernel
#         self.local_kernel_func = local_kernel_func
        

#     def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
#         # left = torch.sqrt(1 / np.exp(self.h) * self.local_kernel_func((x1 - self.x0) / np.exp(self.h)))
#         # right = torch.sqrt(1 / np.exp(self.h) * self.local_kernel_func((self.x0 - x2) / np.exp(self.h)))
        
#         # self.left = torch.sqrt(self.local_kernel_func((x1 - self.x0) / self.h))
#         # self.right = torch.sqrt(self.local_kernel_func((self.x0 - x2) / self.h))
#         # self.base_cov = self.base_kernel(x1, x2, diag, last_dim_is_batch, **params).evaluate()
#         # return self.left.view(-1, 1) * self.base_cov * self.right.view(1,-1)
        
#         left = torch.sqrt(self.local_kernel_func((x1 - self.x0) / self.h))
#         right = torch.sqrt(self.local_kernel_func((self.x0 - x2) / self.h))
#         base_cov = self.base_kernel(x1, x2, diag, last_dim_is_batch, **params).evaluate()
#         return left.view(-1, 1) * base_cov * right.view(1,-1)
    
    
class LocalizedKernel(gpytorch.kernels.Kernel):
    def __init__(self,x_train, h, x0, base_kernel,windowing_func, optimizable_scale=False):
        super().__init__()
        self.x0 = nn.Parameter(x0,requires_grad=False)
        self.logh = nn.Parameter(torch.log(torch.Tensor([h])),requires_grad=optimizable_scale)
        self.base_kernel = base_kernel
        self.windowing_func = windowing_func
        self.x_train = x_train
        
    def forward(self, x1, x2, diag=False, last_dim_is_batch=False, **params):
        # d = x1.shape[1]
        # left = torch.sqrt((1/(self.h**d)) * self.local_kernel_func((x1 - self.x0) / self.h))
        # right = torch.sqrt((1/(self.h**d)) * self.local_kernel_func((self.x0 - x2) / self.h))
        
        h = torch.exp(self.logh)
        left = torch.sqrt(self.windowing_func((x1 - self.x0) / h))
        right = torch.sqrt(self.windowing_func((self.x0 - x2) / h))
        self.left = left
        self.right = right
        self.base_cov = self.base_kernel(x1, x2, diag, last_dim_is_batch, **params).evaluate()
        return left.view(-1, 1) * self.base_kernel(x1, x2, diag, last_dim_is_batch, **params).evaluate() * right.view(1,-1)
    
