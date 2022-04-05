import numpy as np
from kernel_pca import KernelPCA
import matplotlib.pyplot as plt


class KMeans(KernelPCA):

    def __init__(self, dataloader, kernel, n_dim=1, n_cluster=10, display=False):
        super().__init__(dataloader, kernel, n_dim, display)
        self.n_cluster = n_cluster
        self.clusters = None

    def kmeans(self, X, n_iter):
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
        self.PCA(self.dataloader.dataset)
        Z = self.eigenvectors[:self.n_cluster, :]/np.abs(self.eigenvectors[:self.n_cluster, :])
        Z = Z.T
        self.kmeans(Z, n_iter)
        print("accuracy kmeans = ", self.accuracy())
    
    def accuracy(self):
        return np.sum(self.clusters == self.dataloader.target)/self.clusters.shape[0]

# test the kmeans
if __name__ == "__main__":
    mean_1 = np.array([-5, -5])
    mean_2 = np.array([2, 2])
    mean_3 = np.array([1, 1])

    cov_1 = np.array([[1,0], [0,1]])
    cov_2 = np.array([[.1,0], [0,.1]])
    cov_3 = np.array([[3,0], [0,3]])


    test_2 = np.random.multivariate_normal(mean_1, cov_1, size=20)
    test_1 = np.random.multivariate_normal(mean_2, cov_2, size=20)

    test = np.concatenate((test_1, test_2), axis=0)
    mu = np.random.multivariate_normal(mean_3, cov_3, size=2)
    for i in range(20):
        a = np.linalg.norm(mu[None, :, :] - test[:,None,:], axis=2)
        clusters = np.argmin(np.linalg.norm(mu[None, :, :] - test[:,None,:], axis=2), axis=1)
        for k in range(2):
            index = np.argwhere(clusters == k)
            mu[k,:] = np.mean(test[index, :], axis=0)
    x = [[] for i in range(2)]
    y = [[] for i in range(2)]
    for z in range(40):
        x[clusters[z]].append(test[z,0])
        y[clusters[z]].append(test[z,1])

    for i in range(2):
        plt.plot(x[i], y[i], 'x')

    plt.plot(mu[:,0],mu[:,1], 'o')
    plt.show()