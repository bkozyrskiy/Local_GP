import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold

import os, sys
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset


def knn_reg_datasets():
    regr_datasets = ['yacht', 'boston', 'concrete', 'kin8nm' ,'powerplant', 'protein']

    for dataset in ['yacht']:
        print(dataset)
        train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardise=True, random_state=5)
        Xtst, ytst = np.array(test_dataset.tensors[0]),np.array(test_dataset.tensors[1])
        Xtr, ytr = np.array(train_dataset.tensors[0]),np.array(train_dataset.tensors[1])
        # scaler = MinMaxScaler()
        # scaler = StandardScaler()
        # scaler.fit(Xtr)
        # Xtr = scaler.transform(Xtr)
        # Xtst = scaler.fit_transform(Xtst)
        
        parameters = {'n_neighbors':list(range(1, 20, 1))}
        model = KNeighborsRegressor()
        cv = GridSearchCV(model, parameters, scoring='neg_root_mean_squared_error', cv=5)
        cv.fit(Xtr, ytr)
        ytr_pred = cv.predict(Xtr)
        ytst_pred = cv.predict(Xtst)
        print('CV best paramters', cv.best_estimator_)
        print('train MSE', ((ytr_pred - ytr)**2).mean())
        print('test MSE', ((ytst_pred - ytst)**2).mean()) 
        cv_test(np.concatenate((Xtst,Xtr),axis=0), np.concatenate((ytst, ytr),axis=0), cv.best_estimator_.n_neighbors, k_folds=3)
    pass


def cv_test(X, y, n_neighbors, k_folds):
    kf = KFold(n_splits=k_folds)
    fold_mses = []
    for train_index, test_index in kf.split(y):
        Xtr_fold, ytr_fold = X[train_index], y[train_index]
        Xtest_fold, ytest_fold = X[test_index], y[test_index]
        model = KNeighborsRegressor(n_neighbors)
        model.fit(Xtr_fold, ytr_fold)
        ytst_pred = model.predict(Xtest_fold)
        fold_mses.append(((ytst_pred - ytest_fold)**2).mean())
    print('MSE: {}+/-{}'.format(np.mean(fold_mses), np.std(fold_mses)))

if __name__ == '__main__':
    knn_reg_datasets()