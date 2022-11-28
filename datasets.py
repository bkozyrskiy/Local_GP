from torch.utils.data import TensorDataset, DataLoader
import torch
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import json
from config.config import config

def standardize_data(data):
    return (data - data.mean(dim=0))/data.std(dim=0)

def mnist_torch_dataset(batch_size, full_dataset: bool, classes_subset=None):
    '''
    :param batch_size: 
    :param full_dataset - if True, the dataloader will load the full dataset, batch_size ignored 
    :return: 
    '''
    
    default_path = config["default_data_path"]
    train_dataset_path = os.path.join(default_path, 'mnist/pytorch/train_mnist.pth')
    test_dataset_path = os.path.join(default_path, 'mnist/pytorch/test_mnist.pth')

    train_data, train_labels = torch.load(train_dataset_path)
    test_data, test_labels = torch.load(test_dataset_path)
    train_data = train_data.div_(255.).reshape((train_data.shape[0],-1))
    test_data = test_data.div_(255.).reshape((test_data.shape[0],-1))
    if classes_subset is None:
        train_dataset = TensorDataset(train_data, train_labels)
        test_dataset = TensorDataset(test_data, test_labels)
        output_dim = 10
    else:
        tr_indices = np.isin(torch.argmax(train_labels,dim=1), classes_subset).nonzero()[0]
        tst_indices = np.isin(torch.argmax(test_labels,dim=1), classes_subset).nonzero()[0]
        train_dataset = TensorDataset(train_data[tr_indices,...], train_labels[tr_indices,:][:,classes_subset])
        test_dataset = TensorDataset(test_data[tst_indices,...], test_labels[tst_indices,:][:,classes_subset])
        output_dim = len(classes_subset)

    return train_dataset, test_dataset, 784, output_dim

def general_torch_dataset(dataset_name, standardise, standardise_labels=False, random_state=0):
    '''
    :param batch_size: 
    :param full_dataset - if True, the dataloader will load the full dataset, batch_size ignored 
    :return: 
    '''
    
    default_path = config["default_data_path"]
        
    datasets = os.listdir(default_path)
    if dataset_name not in datasets:
        print("Dataset is not available")
        print(datasets)
        raise ValueError()   
    
    path_to_dataset_dir = os.path.join(default_path, dataset_name, 'pytorch')
    # if len(os.listdir(path_to_dataset_dir)) == 2:
    #     train_dataset_path = os.path.join(path_to_dataset_dir, 'train_%s.pth' %dataset_name)
    #     test_dataset_path = os.path.join(path_to_dataset_dir, 'test_%s.pth' %dataset_name)

    #     train_data, train_labels = torch.load(train_dataset_path)
    #     test_data, test_labels = torch.load(test_dataset_path)
        
    # else:
    dataset_path = os.path.join(path_to_dataset_dir, '%s.pth' %dataset_name)
    data, labels = torch.load(dataset_path)
    if standardise:
        data = standardize_data(data)
        
    if standardise_labels:
        scaler = StandardScaler()
        if train_labels.ndim == 1:
            train_labels = train_labels.reshape(-1,1)
            test_labels = test_labels.reshape(-1,1)
            
        train_labels = torch.Tensor(scaler.fit_transform(train_labels))
        test_labels = torch.Tensor(scaler.transform(test_labels))
    
    train_data, test_data, train_labels, test_labels = train_test_split(data,labels, test_size=0.2, random_state=random_state)
        
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    if (len(train_data.shape) != 2) or (len(train_labels.shape) != 2):
        raise ValueError('This loader is not suitable from this dataset') 
    input_dim = train_data.shape[1]
    output_dim = train_labels.shape[1]
    return train_dataset, test_dataset, input_dim, output_dim


def linearly_separable_data(n_train_points, n_test_points, onehot=False):
    x_train = np.random.rand(n_train_points, 2) - 0.5
    x_test = np.random.rand(n_test_points, 2) - 0.5
    y_train = np.apply_along_axis(lambda x: int(x[0] < x[1]), axis=1, arr=x_train)
    y_test = np.apply_along_axis(lambda x: int(x[0] < x[1]), axis=1, arr=x_test)

    if onehot:
        y_train = np.eye(2)[y_train]
        y_test = np.eye(2)[y_test]

    train_dataset = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
    test_dataset = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=n_train_points,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False)
    
    test_dataloader = DataLoader(test_dataset,
                                  batch_size=n_train_points,
                                  shuffle=True,
                                  num_workers=0,
                                  pin_memory=False)

    
    return train_dataloader, test_dataloader
    
def synthetic_regression_problem(train_len, noise_level=0.1):
    '''
    Implements Doppler function from 

    Wasserman, Larry. All of nonparametric statistics. Springer Science & Business Media, 2006.
    Eq. (5.64)
    '''
    x = np.linspace(0,1,1000)
    func = lambda x: np.sqrt(x*(1-x)) * np.sin(2.1*np.pi/(x+0.05))
    train_idx = sorted(np.random.choice(range(len(x)), train_len, replace=False))
    x_train = x[train_idx]
    y_train =(func(x_train) + np.random.randn(len(x_train))*noise_level)
    return x_train.reshape(-1,1), y_train, func