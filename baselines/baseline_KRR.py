import numpy as np
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset

def kernel_regr_reg_datasets():
    # regr_datasets = ['yacht', 'boston', 'concrete', 'energy', 'kin8nm' ,'powerplant', 'protein']
    regr_datasets = ['yacht', 'boston', 'concrete', 'energy']
    # dataset = 'energy'
    for dataset in regr_datasets:
        print(dataset)
        train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=False)
        Xtst, ytst = np.array(test_dataset.tensors[0]),np.array(test_dataset.tensors[1])
        Xtr, ytr = np.array(train_dataset.tensors[0]),np.array(train_dataset.tensors[1])
        scaler = MinMaxScaler()
        # scaler = StandardScaler()
        scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xtst = scaler.fit_transform(Xtst)
        
        parameters = {'alpha':np.logspace(-3, 4, 20), 'gamma':np.logspace(-5, 4, 10)}
        model = KernelRidge(kernel='rbf')
        cv = GridSearchCV(model, parameters, scoring='neg_root_mean_squared_error', cv=5)
        cv.fit(Xtr, ytr)
        ytr_pred = cv.predict(Xtr)
        ytst_pred = cv.predict(Xtst)
        print('CV best paramters', cv.best_estimator_)
        print('train MSE', ((ytr_pred - ytr)**2).mean())
        print('test MSE', ((ytst_pred - ytst)**2).mean())
    pass

if __name__ == '__main__':
    kernel_regr_reg_datasets()