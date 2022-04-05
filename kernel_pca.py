import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt

class KernelPCA:

    def __init__(self, dataloader, kernel, n_dim, display=False):
        self.dataloader = dataloader.copy_data()

        # kernel should be a function here
        self.kernel = kernel
        self.n_dim = n_dim
        self.display = display
        self.eigenvectors = None
        self.eigenvalues = None
    
    def center_kernel(self, X):
        """center kernel (mean) before pca
        """
        N, _ = X.shape
        kernel_mat = self.kernel(X)
        ones_N = 1/N*np.ones((N, N))
        return kernel_mat - ones_N@kernel_mat - kernel_mat@ones_N + ones_N@kernel_mat@ones_N
    
    def PCA(self, X):
        """compute pca
        """
        K = self.center_kernel(X)
        eigenvals, eigenvects = eigh(K)

        # remove the eigenvectors associated to 0
        # it's a sanity check, should never occur
        non_zero = eigenvals != 0
        eigenvals = eigenvals[non_zero]
        eigenvects = eigenvects[non_zero, :]

        # sort eigenvalues in descending order to retrieve 
        # the first n_dim eigenvect
        order_eigh = eigenvals.argsort()[::-1]
        
        if self.display:
            plt.bar(np.arange(len(order_eigh)), np.abs(eigenvals[order_eigh]))
            plt.show()
        # store for the kernelmeans
        self.eigenvectors = eigenvects[order_eigh, :]
        self.eigenvalues = eigenvals[order_eigh]
        order_eigh = order_eigh[:self.n_dim]
        alpha = eigenvects[order_eigh, :].T/np.sqrt(eigenvals[order_eigh])
        return np.dot(K, alpha)

    def project(self):
        """return the result of the pca projection
        """
        N, _ = self.dataloader.dataset.shape
        if self.dataloader.validate_set is not None:
            dataset_plain = np.concatenate([self.dataloader.dataset, self.dataloader.validate_set], axis=0)
        else:
            dataset_plain = self.dataloader.dataset
        dataset_reduced = self.PCA(dataset_plain)
        self.dataloader.dataset = dataset_reduced[:N,:]
        self.dataloader.validate_set =  dataset_reduced[N:,:]
        self.dataloader.K = self.dataloader.kernel(self.dataloader.dataset_train)
        return self.dataloader
        