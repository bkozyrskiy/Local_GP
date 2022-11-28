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
from utils import get_hs, n_neighbours_to_h, get_effective_datapoints
from kernels import hilbert_kernel, epanechnikov_windowing_func




def cv_params_search(Xtr, ytr, params, k=3, max_n_points_ratio=0.2, windowing_func=None):
    '''
        params 
    '''
    def cv_kernel_params_search(Xtr, ytr, params, n, k_folds, max_n_points):
        best_val_mse = np.inf
        for alpha in params["alpha"]:
            for kernel in params["kernel"]:
                fold_val_mse = 0
                kf = KFold(n_splits=k_folds)
                for train_index, val_index in kf.split(ytr):
                    Xtr_fold, ytr_fold = Xtr[train_index], ytr[train_index]
                    Xval_fold, yval_fold = Xtr[val_index], ytr[val_index]
                    center_mse = 0
                    for idx_center in range(Xval_fold.shape[0]):
                        # w = hilbert_kernel((Xtr_fold - Xval_fold[idx_center])/h)
                        h = n_neighbours_to_h(Xval_fold[idx_center], Xtr_fold, n_points=n)
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
                val_mse = fold_val_mse/k_folds
                if best_val_mse > val_mse:
                    best_val_mse = val_mse
                    best_kernel_parameters = {"alpha":alpha, "kernel":kernel}
        return best_kernel_parameters, best_val_mse
    
    best_cv_mse = np.inf
    max_n_points = max_n_points_ratio * Xtr.shape[0]
    for n in params["n_points"]:
        print('n_points', n)
        try:
            n_kernel_parameters, n_cv_mse = cv_kernel_params_search(Xtr, ytr, params, n, k, max_n_points)
        except ValueError as inst:
            if inst.args[0] == 'h is too small':
                continue
            elif inst.args[0] == 'h is too big':
                break
            else:
                raise inst
        
        if best_cv_mse > n_cv_mse:
            best_cv_mse = n_cv_mse
            best_parameters = {**n_kernel_parameters, **{'n_points':n}, **{'val mse':n_cv_mse}}
            print("Candidate parameters", best_parameters)                
    return best_cv_mse, best_parameters


def test(Xtr, ytr, Xtst, ytst, params, windowing_func):
    n = params["n_points"]
    kernel = params['kernel']
    alpha = params['alpha']
    center_mse = 0
    for idx_center in range(ytst.shape[0]):
        h = n_neighbours_to_h(Xtst[idx_center], Xtr, n_points=n)
        w = windowing_func((Xtr - Xtst[idx_center])/h)
        model = KernelRidge(kernel=kernel, alpha=alpha)
        Xtr_nonzero, ytr_nonzero, w_nonzero = get_effective_datapoints(Xtr, ytr, w)
        model.fit(Xtr_nonzero, ytr_nonzero, sample_weight=w_nonzero)
        y_center_pred = model.predict(Xtst[idx_center].reshape(1,-1))
        center_mse += (y_center_pred - ytst[idx_center])**2
    center_mse = center_mse/len(ytst)
    return center_mse

if __name__ == "__main__":
    # datasets = ['yacht', 'boston','concrete', 'kin8nm', 'powerplant', 'protein']
    datasets = ['protein']
    for dataset in datasets:
        print(dataset)
        train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=False,random_state=0)
        Xtr, ytr = np.array(train_dataset.tensors[0]), np.array(train_dataset.tensors[1])
        Xtst, ytst = np.array(test_dataset.tensors[0]), np.array(test_dataset.tensors[1])
        scaler = MinMaxScaler()
        
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xtst = scaler.fit_transform(Xtst)
        
        # params = {"alpha": np.linspace(0.0001,0.1,10),
        #         "kernel": [RBF(l) for l in np.linspace(0.05,10,10)],
        #         "n_points": [100]}
        ###################################################################
        #boston
        # params = {
        #     "alpha": [0.0001],
        #     "kernel": [RBF(l) for l in [10]],
        #     "n_points": [9]
        # }
        ####################################################################
        # params = {"alpha": [0.001],
        #         "kernel": [RBF(l) for l in np.linspace(0.05,10,50)],
        #         "n_points": [10,30,50,70,90]}
        # best_val_mse, best_parameters = cv_params_search(Xtr, ytr, params, k=3, windowing_func=epanechnikov_windowing_func)
        best_parameters = {'alpha': 0.0001, 'kernel': RBF(length_scale=10), 'n_points': 60}
        
        center_mse = test(Xtr, ytr, Xtst, ytst, best_parameters, windowing_func=hilbert_kernel)
        print('Test mse {}'.format(center_mse))
        print(best_parameters)