import numpy as np
import os, sys
from sklearn.kernel_ridge import KernelRidge
import sklearn
from sklearn.model_selection import KFold

from sklearn.gaussian_process.kernels import RBF
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from multiprocessing import Pool
from functools import partial
import gc


PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset
from utils import get_hs, n_neighbours_to_h, get_effective_datapoints
from kernels import hilbert_kernel, epanechnikov_windowing_func

# def cv_fold(Xtr_fold, ytr_fold, Xval_fold, yval_fold, alpha,kernel, n, max_n_points, windowing_func):
#     center_mse = 0
#     for idx_center in range(Xval_fold.shape[0]):
#         # w = hilbert_kernel((Xtr_fold - Xval_fold[idx_center])/h)
#         h = n_neighbours_to_h(Xval_fold[idx_center], Xtr_fold, n_points=n)
#         w = windowing_func((Xtr_fold - Xval_fold[idx_center])/h)
#         if np.all(w == 0):
#             print('Too few points')
#             # raise ValueError('h is too small')
#             return None
#         if (w != 0).sum() > max_n_points:
#             print('Too many points')
#             # raise ValueError('h is too big')
#             return None 
#         model = KernelRidge(kernel=kernel, alpha=alpha)
#         Xtr_fold_nonzero, ytr_fold_nonzero, w_nonzero = get_effective_datapoints(Xtr_fold, ytr_fold, w)
#         model.fit(Xtr_fold_nonzero, ytr_fold_nonzero, sample_weight=w_nonzero)
#         y_center_pred = model.predict(Xval_fold[idx_center].reshape((1,-1)))
#         center_mse += (y_center_pred - yval_fold[idx_center]) ** 2
#     return center_mse

def local_krr(Xtr, ytr, Xcenter, ycenter, alpha, kernel, max_n_points, n, windowing_func):
    h = n_neighbours_to_h(Xcenter, Xtr, n_points=n)
    w = windowing_func((Xtr - Xcenter)/h)
    if np.all(w == 0):
        print('Too few points')
        # raise ValueError('h is too small')
        return None
    if (w != 0).sum() > max_n_points:
        print('Too many points')
        # raise ValueError('h is too big')
        return None 
    model = KernelRidge(kernel=kernel, alpha=alpha)
    Xtr_fold_nonzero, ytr_fold_nonzero, w_nonzero = get_effective_datapoints(Xtr, ytr, w)
    model.fit(Xtr_fold_nonzero, ytr_fold_nonzero, sample_weight=w_nonzero)
    y_center_pred = model.predict(Xcenter.reshape((1,-1)))
    return (y_center_pred - ycenter) ** 2


def cv_kernel_params_search(params, Xtr, ytr, n, k_folds, max_n_points, windowing_func):
        alpha = params[0]
        kernel = params[1]
        kf = KFold(n_splits=k_folds)
        fold_val_mse = 0
        for train_index, val_index in kf.split(ytr):
            Xtr_fold, ytr_fold = Xtr[train_index], ytr[train_index]
            Xval_fold, yval_fold = Xtr[val_index], ytr[val_index]
            center_mse = 0
            for idx_center in range(Xval_fold.shape[0]):
                # h = n_neighbours_to_h(Xval_fold[idx_center], Xtr_fold, n_points=n)
                # w = windowing_func((Xtr_fold - Xval_fold[idx_center])/h)
                # if np.all(w == 0):
                #     print('Too few points')
                #     # raise ValueError('h is too small')
                #     return None
                # if (w != 0).sum() > max_n_points:
                #     print('Too many points')
                #     # raise ValueError('h is too big')
                #     return None 
                # model = KernelRidge(kernel=kernel, alpha=alpha)
                # Xtr_fold_nonzero, ytr_fold_nonzero, w_nonzero = get_effective_datapoints(Xtr_fold, ytr_fold, w)
                # model.fit(Xtr_fold_nonzero, ytr_fold_nonzero, sample_weight=w_nonzero)
                # y_center_pred = model.predict(Xval_fold[idx_center].reshape((1,-1)))
                # center_mse += (y_center_pred - yval_fold[idx_center]) ** 2
                
                local_mse = local_krr(Xtr_fold, ytr_fold, Xval_fold[idx_center], yval_fold[idx_center], alpha, kernel, max_n_points, n, windowing_func)
                if local_mse is None:
                    return None
                else:
                    center_mse += local_mse
                gc.collect()
            fold_val_mse += center_mse/Xval_fold.shape[0]
            gc.collect()
            
        val_mse = fold_val_mse/k_folds
        return val_mse
    
def cv_params_search(Xtr, ytr, params_range, k=3, max_n_points_ratio=0.2, windowing_func=None):
    best_cv_mse = np.inf
    
    krr_params = [(alpha, kernel) for alpha in params_range['alpha'] for kernel in params_range['kernel'] ]
    max_n_points = max_n_points_ratio * Xtr.shape[0]
    
    for n in params_range["n_points"]:
        print('n_points', n)
        f = partial(cv_kernel_params_search, Xtr=Xtr, ytr=ytr, n=n, k_folds=k, max_n_points=max_n_points, windowing_func=windowing_func)
        p = Pool(processes=8)
        val_mses = p.map(f, krr_params)
        if None in val_mses:
            continue
        best_idx = np.argmin(val_mses)
        if val_mses[best_idx] < best_cv_mse:
            best_cv_mse = val_mses[best_idx]
            best_parameters = {**{'alpha': krr_params[best_idx][0]},**{'kernel': krr_params[best_idx][1]}, **{'n_points':n}, **{'val mse':best_cv_mse}}
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
    datasets = ['kin8nm']
    for dataset in datasets:
        print(dataset)
        train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=False)
        Xtr, ytr = np.array(train_dataset.tensors[0]), np.array(train_dataset.tensors[1])
        Xtst, ytst = np.array(test_dataset.tensors[0]), np.array(test_dataset.tensors[1])
        scaler = MinMaxScaler()
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xtst = scaler.fit_transform(Xtst)
        params = {"alpha": np.linspace(0.0001,0.1, 10),
                "kernel": [RBF(l) for l in np.linspace(0.05,10,10)],
                "n_points": range(3,10, 2)}
        best_val_mse, best_parameters = cv_params_search(Xtr, ytr, params, k=3, windowing_func=epanechnikov_windowing_func)
        
        ####################################################
        #Protein. Epanechnikov for testing
        # best_parameters = {'alpha': 0.0001, 
        #                    'kernel': RBF(length_scale=10), 
        #                    'n_points': 20}
        
        center_mse = test(Xtr, ytr, Xtst, ytst, best_parameters, windowing_func=epanechnikov_windowing_func)
        print('Test mse {}'.format(center_mse))
        print(best_parameters)