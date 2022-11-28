import os, sys
from sklearn.kernel_ridge import KernelRidge
import sklearn
from sklearn.model_selection import KFold
import numpy as np
from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import math
from tqdm import tqdm

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset
from utils import get_hs

def hilbert_kernel(x):
    if len(x.shape) == 2:
        return (1 / np.linalg.norm(x,axis=-1))* (np.linalg.norm(x,axis=-1) <= 1)
    if len(x.shape) == 1:
        return 1 / np.abs(x)* (np.abs(x) <= 1)
    
def epanechnikov_windowing_func(x):
    def unit_sphere_V(d):
        return (np.pi**(d/2)/math.gamma(1 + d/2))
    
    if len(x.shape) == 2:
        num = x.shape[1]+2
        denum = 2 * unit_sphere_V(x.shape[1])
        return (num/denum) * (1 - np.linalg.norm(x,axis=1)**2) * (np.linalg.norm(x,axis=1) <= 1)
       
    if len(x.shape) == 1:
        return (3/4) * (1 - np.linalg.abs(x)) * (np.linalg.abs(x) <= 1)

def get_effective_datapoints(X,y,w):
    nonzero_idx = w.nonzero()[0]
    return X[nonzero_idx,:], y[nonzero_idx], w[nonzero_idx]

def cv_params_search(Xtr, ytr, params, k=3, max_n_points_ratio=0.2, windowing_func=None):
    '''
        params 
    '''
    def cv_kernel_params_search(Xtr, ytr, params, h, k, max_n_points):
        best_val_mse = np.inf
        for alpha in params["alpha"]:
            for kernel in params["kernel"]:
                kf = KFold(n_splits=k)
                for train_index, val_index in kf.split(ytr):
                    Xtr_fold, ytr_fold = Xtr[train_index], ytr[train_index]
                    Xval_fold, yval_fold = Xtr[val_index], ytr[val_index]
                    fold_val_mse = 0
                    center_mse = 0
                    for idx_center in range(Xval_fold.shape[0]):
                        w = windowing_func((Xtr_fold - Xval_fold[idx_center])/h)
                        if np.all(w == 0):
                            print('Too few points')
                            raise ValueError('h is too small')
                        if (w != 0).sum() > max_n_points:
                            print('Too many points')
                            raise ValueError('h is too big') 
                        model = KernelRidge(kernel=kernel, alpha=alpha)
                        Xtr_fold_nonzero, ytr_fold_nonzero, w_nonzero = get_effective_datapoints(Xtr_fold, ytr_fold, w)
                        model.fit(Xtr_fold_nonzero, ytr_fold_nonzero, sample_weight=w_nonzero)
                        y_center_pred = model.predict(Xval_fold[idx_center].reshape((1,-1)))
                        center_mse += (y_center_pred - yval_fold[idx_center]) ** 2
                    fold_val_mse += center_mse/Xval_fold.shape[0]
                val_mse = fold_val_mse/k
                if best_val_mse > val_mse:
                    best_val_mse = val_mse
                    best_kernel_parameters = {"alpha":alpha, "kernel":kernel}
        return best_kernel_parameters, best_val_mse
    
    best_cv_mse = np.inf
    max_n_points = max_n_points_ratio * Xtr.shape[0]
    for h in params["h"]:
        print('h', h)
        try:
            h_kernel_parameters, h_cv_mse = cv_kernel_params_search(Xtr, ytr, params, h, k, max_n_points)
        except Exception as inst:
            if inst.args[0] == 'h is too small':
                continue
            elif inst.args[0] == 'h is too big':
                break
            else:
                raise inst
            
        
        if best_cv_mse > h_cv_mse:
            best_cv_mse = h_cv_mse
            best_parameters = {**h_kernel_parameters, **{'h':h}, **{'val mse':h_cv_mse}}
            print("Candidate parameters", best_parameters)                
    return best_cv_mse, best_parameters


def test(Xtr, ytr, Xtst, ytst, params, windowing_func):
    h = params["h"]
    kernel = params['kernel']
    alpha = params['alpha']
    center_mse = 0
    for idx_center in range(ytst.shape[0]):
        w = windowing_func((Xtr - Xtst[idx_center])/h)
        Xtr_nonzero, ytr_nonzero, w_nonzero = get_effective_datapoints(Xtr, ytr, w)
        model = KernelRidge(kernel=kernel, alpha=alpha)
        model.fit(Xtr_nonzero, ytr_nonzero, sample_weight=w_nonzero)
        y_center_pred = model.predict(Xtst[idx_center].reshape(1,-1))
        center_mse += (y_center_pred - ytst[idx_center])**2
    center_mse = center_mse/len(ytst)
    return center_mse

if __name__ == "__main__":
    # datasets = ['yacht', 'boston','concrete', 'kin8nm', 'powerplant', 'protein']
    datasets = ['concrete']
    for dataset in datasets:
        print(dataset)
        train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=False)
        Xtr, ytr = np.array(train_dataset.tensors[0]), np.array(train_dataset.tensors[1])
        Xtst, ytst = np.array(test_dataset.tensors[0]), np.array(test_dataset.tensors[1])
        # scaler = MinMaxScaler()
        scaler = StandardScaler()
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xtst = scaler.fit_transform(Xtst)
        hs = get_hs(Xtst, Xtr, min_points=3)
        
        #???
        # params = {"alpha": np.linspace(0.001,0.1,10),
        #         "kernel": [RBF(l) for l in np.linspace(0.1,10,10)],
        #         "h": np.linspace(0.18, 0.188, 1000)}
        # yacht
        # params = {"alpha": np.linspace(0.0001,0.1,10),
        #         "kernel": [RBF(l) for l in np.linspace(0.1,10,50)],
        #         "h": np.linspace(0.60, 0.70, 10)}
        # params = {"alpha": np.linspace(0.0001,0.1,10),
        #         "kernel": [RBF(l) for l in np.linspace(0.1,10,50)],
        #         "h": np.linspace(hs[0], hs[-1], 10)}
        ####################################################################
        #boston
        # params = {"alpha": np.linspace(0.0001,0.1,10),
        #         "kernel": [RBF(l) for l in np.linspace(0.1,10,50)],
        #         "h": np.linspace(hs[0], hs[-1], 100)}
        #####################################################################
        #concrete 
        params = {"alpha": np.linspace(0.0001,0.4,15),
                "kernel": [RBF(l) for l in np.linspace(0.05,10,50)],
                "h": hs} 
        # params = {"alpha": [0.0001],
        #         "kernel": [RBF(l) for l in [1.67]],
        #         "h": [0.46]}
         
        best_val_mse, best_parameters = cv_params_search(Xtr, ytr, params, k=3, windowing_func=epanechnikov_windowing_func)
        center_mse = test(Xtr, ytr, Xtst, ytst, best_parameters,windowing_func=epanechnikov_windowing_func)
        print('Mean center mse {}'.format(center_mse))
        print(best_parameters)