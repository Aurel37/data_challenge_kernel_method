import numpy as np

class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
        #self.k = None
        
    def kernel(self,X,Y = None):
        ## Input vectors X and Y of shape Nxd and Mxd
        # difference between all vectors X and Y

        if Y is None:
            N, d = X.shape
            norm_diff = np.zeros((N, N))
            for i in range(N):
                if i%100 == 0:
                    print(i/N)
                for j in range(i + 1, N):
                    norm_diff[i, j] = np.linalg.norm(X[i, :] - X[j, :])
            norm_diff = norm_diff + norm_diff.T
        else:
            N, d = X.shape
            M, dy = Y.shape
            if not d == dy:
                raise ValueError("X and Y don't have the same dimension ")
            norm_diff = np.zeros((N, M))
            for i in range(N):
                for j in range(M):
                    norm_diff[i, j] = np.linalg.norm(X[i, :] - Y[j, :])
        return np.exp(-1/2*np.square(norm_diff/self.sigma))

class Linear:
    def __init__(self):
        self.k = None
    
    def kernel(self,X,Y=None):
        # matrixes of inner products
        if Y is not None:
            x_tens_y = X[:,:]@Y[:, :].T
            self.k = x_tens_y
            return x_tens_y
        else:
            x_tens_y = X[:,:]@X[:, :].T
            self.k = x_tens_y
            return x_tens_y