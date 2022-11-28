import os, sys
from sklearn.kernel_ridge import KernelRidge
import sklearn 
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import MinMaxScaler

import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset

def kernel_rff_regr_reg_datasets():
    # regr_datasets = ['kin8nm' ,'powerplant', 'protein']
    regr_datasets = ['kin8nm', 'powerplant', 'protein']
    # dataset = 'energy'
    for dataset in regr_datasets:
        print(dataset)
        train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=False)
        Xtst, ytst = np.array(test_dataset.tensors[0]),np.array(test_dataset.tensors[1])
        Xtr, ytr = np.array(train_dataset.tensors[0]),np.array(train_dataset.tensors[1])
        scaler = MinMaxScaler()
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xtst = scaler.fit_transform(Xtst)
        # feature_map = Nystroem(kernel='rbf', n_components=1000, gamma=.2, random_state=1)
        feature_map = RBFSampler(gamma=1,n_components=4000, random_state=1)
        # parameters = {'alpha':np.logspace(-3, 4, 20), 'gamma':np.logspace(-5, 4, 10)}
        clf = KernelRidge(alpha=0.01, kernel='linear')
        pipe = Pipeline(steps=[('feature_map', feature_map), ('clf', clf)])
        param_grid = {
            'feature_map__gamma': np.logspace(-7, 6, 10),
            'clf__alpha': np.logspace(-6, 2, 10),
        }
        # search = RandomizedSearchCV(pipe, param_grid,scoring='neg_root_mean_squared_error', n_iter=30, n_jobs=1, verbose=2)
        search = GridSearchCV(pipe, param_grid, scoring='neg_root_mean_squared_error', cv=5)
        search.fit(Xtr, ytr)
        ytr_pred = search.best_estimator_.predict(Xtr)
        ytst_pred = search.best_estimator_.predict(Xtst)
        print('CV best paramters', search.best_estimator_)
        print('train MSE', ((ytr_pred - ytr)**2).mean())
        print('test MSE', ((ytst_pred - ytst)**2).mean())
    pass

if __name__ == '__main__':
    kernel_rff_regr_reg_datasets()