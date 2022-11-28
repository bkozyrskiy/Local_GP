import os, sys
from sklearn.kernel_ridge import KernelRidge
import sklearn 
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.gaussian_process import GaussianProcessClassifier, GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from datasets import general_torch_dataset

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

def kernel_regr_reg_datasets():
    dataset = 'powerplant'
    regr_datasets = ['yacht', 'boston', 'concrete', 'energy', 'kin8nm', 'naval','powerplant', 'protein']
    train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardize=True)
    Xtst, ytst = np.array(test_dataset.tensors[0]),np.array(test_dataset.tensors[1])
    Xtr, ytr = np.array(train_dataset.tensors[0]),np.array(train_dataset.tensors[1])
    # opu_kernel = lambda x,y: (np.linalg.norm(x,ord=2)**2) * (np.linalg.norm(x,ord=2)**2) + x.dot(y)**2 
    # rbf_kernel = lambda x,y: np.exp(-(1.0/len(x)) * np.linalg.norm(x-y)**2)
    # clf = KernelRidge(alpha=0.01, kernel='rbf')
    clf = KernelRidge(alpha=0.01, kernel='rbf')
    clf.fit(Xtr, ytr)
    ytr_pred = clf.predict(Xtr)
    ytst_pred = clf.predict(Xtst)
    print(((ytr_pred - ytr)**2).mean())
    print(((ytst_pred - ytst)**2).mean())
    pass

def kernel_regr_clf_datasets():
    # regr_datasets = ['yacht', 'boston', 'concrete', 'energy', 'kin8nm', 'naval','powerplant', 'protein']
    # clf_datasets = ['eeg', 'magic', 'miniboo', 'letter', 'drive', 'mocap']
    for dataset in ['eeg', 'magic', 'letter', 'drive']:
        train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardize=True)
        Xtst, ytst = np.array(test_dataset.tensors[0]),np.array(test_dataset.tensors[1])
        Xtr, ytr = np.array(train_dataset.tensors[0]),np.array(train_dataset.tensors[1])
        ytr[ytr==0], ytst[ytst==0] = -1, -1
        # opu_kernel = lambda x,y: (np.linalg.norm(x,ord=2)**2) * (np.linalg.norm(x,ord=2)**2) + x.dot(y)**2 
        # rbf_kernel = lambda x,y: np.exp(-(1.0/len(x)) * np.linalg.norm(x-y)**2)
        # clf = KernelRidge(alpha=0.01, kernel='rbf')
        clf = KernelRidge(alpha=0.01, kernel='rbf')
        parameters = {'alpha':[0, 0.01, 0.1, 0.3, 0.5]}
        gs = GridSearchCV(clf, parameters, scoring='neg_mean_squared_error', cv=3)
        gs.fit(Xtr,ytr)
        ytr_pred = gs.best_estimator_.predict(Xtr)
        ytst_pred = gs.best_estimator_.predict(Xtst)
        # clf.fit(Xtr, ytr)
        # ytr_pred = clf.predict(Xtr)
        # ytst_pred = clf.predict(Xtst)
        print('Dataset: {}, tr_error: {}, tst_error: {}, alpha :{}'.format(
            dataset,
            (np.argmax(ytr_pred,axis=1) != np.argmax(ytr,axis=1)).mean(), 
            (np.argmax(ytst_pred,axis=1) != np.argmax(ytst,axis=1)).mean(),
            gs.best_estimator_.alpha))


def fake_optimizer(obj_func, initial_theta, bounds):
    # * 'obj_func' is the objective function to be minimized, which
    #   takes the hyperparameters theta as parameter and an
    #   optional flag eval_gradient, which determines if the
    #   gradient is returned additionally to the function value
    # * 'initial_theta': the initial value for theta, which can be
    #   used by local optimizers
    # * 'bounds': the bounds on the values of theta
    # Returned are the best found hyperparameters theta and
    # the corresponding value of the target function.
    theta_opt = initial_theta
    func_min = -np.inf
    return theta_opt, func_min

def gp_regr_regr_datasets():
    regr_datasets = ['yacht', 'boston', 'concrete', 'energy', 'kin8nm', 'powerplant', 'protein']
    for dataset  in regr_datasets:
        train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardize=True)
        Xtst, ytst = np.array(test_dataset.tensors[0]),np.array(test_dataset.tensors[1])
        Xtr, ytr = np.array(train_dataset.tensors[0]),np.array(train_dataset.tensors[1])
        kernel = RBF(1)
        # model = GaussianProcessRegressor(kernel=kernel, optimizer=fake_optimizer, n_restarts_optimizer=0, alpha=1e-10)
        model = GaussianProcessRegressor(kernel=kernel, optimizer="fmin_l_bfgs_b", n_restarts_optimizer=0, alpha=1)
        model.fit(Xtr, ytr)
        # model.alpha_ = np.ones_like(model.alpha_)*0
        ytr_pred = model.predict(Xtr)
        ytst_pred = model.predict(Xtst)
        train_mse = ((ytr_pred - ytr)**2).mean()
        test_mse = ((ytst_pred - ytst)**2).mean()
        print('Dataset: {}, alpha: {}, train_mse: {}, test_mse: {}'.format(dataset, model.alpha, train_mse, test_mse))
        # parameters = {'alpha':[0, 0.01, 0.1, 0.3, 0.5]}
        # gs = GridSearchCV(model, parameters, scoring='neg_mean_squared_error', cv=5)
        # gs.fit(Xtr,ytr)
        # ytr_pred = gs.best_estimator_.predict(Xtr)
        # ytst_pred = gs.best_estimator_.predict(Xtst)
        # train_mse = ((ytr_pred - ytr)**2).mean()
        # test_mse = ((ytst_pred - ytst)**2).mean()
        # print('Dataset: {}, alpha: {}, train_mse: {}, test_mse: {}'.format(dataset, gs.best_estimator_.alpha, train_mse, test_mse))
        pass
    


if __name__ == '__main__':
    gp_regr_regr_datasets()