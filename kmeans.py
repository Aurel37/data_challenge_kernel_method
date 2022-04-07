import numpy as np
from kernel_pca import KernelPCA
import matplotlib.pyplot as plt


class KMeans(KernelPCA):
    """Kmeans algorithms, inherit from KernelPCA using the kernel "kernel"

    Methods:
        kmeans : perform the kmeans algorithms
        spectral_clustering : perform the kernel kmeans algo 
        using spectral clustering
        accuracy : compute accuracy

    Attributes:
        dataloader : np.array
        kernel : func
        n_dim : float (useless here for constitency)
        n_cluster : int
        display : bool (useless here for constitency)
    """

    def __init__(self, dataloader, kernel, n_dim=1, n_cluster=10, display=False):
        super().__init__(dataloader, kernel, n_dim, display)
        self.n_cluster = n_cluster
        self.clusters = None

    def kmeans(self, X, n_iter):
        """classic kmeans algorithms for the dataset X

        Parameters:
            X : np.array
            n_iter : int
        
        Return:
            mu : np.array, then means found by kmeans
            clusters : np.array the clusters assigned to each vectors
        """
        N, d = X.shape
        mu = np.random.multivariate_normal(np.mean(X, axis=0), np.eye(self.n_cluster), size=self.n_cluster)
        for i in range(n_iter):
            clusters = np.argmax(np.linalg.norm(mu[None, :, :] - X[:,None,:], axis=2), axis=1)
            for k in range(self.n_cluster):
                index = np.argwhere(clusters == k)
                if index != []:
                    mu[k,:] = np.mean(X[index, :], axis=0)
        self.clusters = clusters
        return mu, clusters + 1

    def spectral_cluestering(self, n_iter):
        """perform the kernel kmeans algo 
        using spectral clustering and compute
        accuracy

        Parameters:
            n_iter : int
        """
        self.PCA(self.dataloader.dataset)
        Z = self.eigenvectors[:self.n_cluster, :]/np.abs(self.eigenvectors[:self.n_cluster, :])
        Z = Z.T
        self.kmeans(Z, n_iter)
        print("accuracy kmeans = ", self.accuracy())
    
    def accuracy(self):
        """Compute accuracy for the dataloader
        """
        return np.sum(self.clusters == self.dataloader.target)/self.clusters.shape[0]
