import numpy as np
from scipy.linalg import eigh

class KernelPCA:

    def __init__(self, dataloader, kernel, n_dim):
        self.dataloader = dataloader.copy()

        # kernel should be a function here
        self.kernel = kernel
        self.n_dim = n_dim
    
    def center_kernel(self, X):
        """center kernel (mean) before pca
        """
        N, _ = X.shape
        kernel_mat = self.kernel(X, X)
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
        order_eigh = order_eigh[:self.n_dim]

        alpha = eigenvects[order_eigh, :].T/np.sqrt(eigenvals[order_eigh])
        return np.dot(K, alpha)

    def project(self):
        """return the result of the pca projection
        """
        self.dataloader.dataset = self.PCA(self.dataloader.dataset)
        self.dataloader.K = self.kernel(self.dataloader.dataset_train, self.dataloader.dataset_train)
        self.dataloader.kernel = self.kernel
        return self.dataloader
        