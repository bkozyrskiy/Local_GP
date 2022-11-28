import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
import torch

# def get_hs(x_test, x_train):
#     distances = pairwise_distances(x_test, x_train)
#     hs = sorted(set(distances.reshape(-1)))
#     return hs

def get_h_limits(x_test, x_train):
    distances = pairwise_distances(x_test, x_train)
    np.sort(distances, axis=1) #inplace operation
    min_h = distances[:,0].max()
    max_h = distances[:,0].max()
    return (min_h, max_h)

def get_hs(x_test, x_train, min_points, step_points=1):
    '''
    min_points: int, ensures, that for each point from x_test it choises at least min_points
    step_points: int, ensures, that on each step of h will be added at leat step_points points
    '''
    eps = 1e-15
    distances = pairwise_distances(x_test, x_train) + eps
    distances = np.sort(distances, axis=1)
    min_h = distances[:,min_points-1].max()
    flattened = distances.reshape(-1)
    hs = sorted(set(flattened[flattened >= min_h]))
    return hs

def n_neighbours_to_h(central_point, x_train, n_points):
    # eps = 1e-15
    eps = 1e-10
    if central_point.ndim == 1:
        central_point = central_point.reshape(1, -1)
    distances = pairwise_distances(central_point, x_train) + eps
    distances = np.sort(distances)
    return distances[0,n_points-1]

def get_effective_datapoints(X,y,w):
    if isinstance(w,np.ndarray):
        nonzero_idx = w.nonzero()[0]
    elif isinstance(w, torch.Tensor):
        nonzero_idx = w.nonzero()[:,0]
    return X[nonzero_idx,:], y[nonzero_idx], w[nonzero_idx]

def standardise_labels(y):
    if isinstance(y, torch.Tensor):
        mean = torch.mean(y, axis=0)
        std = torch.std(y, axis=0)
    else:
        mean = np.mean(y, axis=0)
        std = np.std(y, axis=0)
    y_normed = (y - mean)/std
    return y_normed, mean, std
    
if __name__ == '__main__':
    # np.random.seed(2) #7,8
    a = np.random.rand(3,6)
    b = np.random.rand(10,6)
    d = pairwise_distances(a, b)
    hs = get_hs(a, b, min_points=1)
    pass