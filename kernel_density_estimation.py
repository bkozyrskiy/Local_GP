from datasets import synthetic_regression_problem, general_torch_dataset
from sklearn.neighbors import KernelDensity
import numpy as np
from sklearn.neighbors.tests.test_kde import compute_kernel_slow

if __name__ == '__main__':
    dataset = 'yacht'
    train_dataset, test_dataset, input_dim, output_dim = general_torch_dataset(dataset, standardize=True)
    x_train, y_train = train_dataset.tensors[0].numpy(), train_dataset.tensors[1].squeeze().numpy()
    x_test, y_test = test_dataset.tensors[0].numpy(), test_dataset.tensors[1].squeeze().numpy()
    # compute_kernel_slow(x_train, x_test[:1],kernel='tophat', h=np.exp(0))
    
    for idx in range(x_test.shape[0]):
        x = x_test[idx,:].reshape((1,-1))
        loghs = np.arange(-1,5,0.5)
        klds = []
        for logh in loghs:
            # print("Log h",logh)
            h = np.exp(logh)
            kern_debug = compute_kernel_slow(x_train, x, kernel='tophat', h=h)
            kde = KernelDensity(kernel='tophat', bandwidth=h).fit(x_train)
            klds.append(kde.score_samples(x))
        print(loghs[np.argmax(klds)])
        # mkld = kde.score_samples(x_test).mean()
        # print('Mean kernel log-density',mkld)
        
        
    