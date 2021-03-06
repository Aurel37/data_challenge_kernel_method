"""All  kernel used in the competition are encoded here
Gaussian kernel/Polynomial Kernel
"""
import numpy as np

class RBF:
    """Polynomial kernel (x.T@x + c)^d

    Methods:
        kernel : compute the kernel

    Attributes:
        k : None, here for constitency
        sigma : float
    """

    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
        #self.k = None
        
    def kernel(self,X,Y = None):
        """Compute the kernel k(X,Y) 
        if Y is not None, k(X,X) otherwise

        Parameters:
            X : np.array
            Y : np.array
        
        Return:
            a gaussian kernel in an array
        """
        ## Input vectors X and Y of shape Nxd and Mxd
        # difference between all vectors X and Y

        if Y is None:
            N, d = X.shape
            norm_diff = np.zeros((N, N))
            for i in range(N):
                if i%100 == 0:
                    print(f"\r current = {i/N}", end="")
                for j in range(i + 1, N):
                    norm_diff[i, j] = np.linalg.norm(X[i, :] - X[j, :])
            print()
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


class Polynomial:
    """Polynomial kernel (x.T@x + c)^d

    Methods:
        kernel : compute the kernel

    Attributes:
        k : None, here for constitency
        d : float 
        c : float
    """

    def __init__(self, d = 2, c = 0):
        self.k = None
        self.d = d
        self.c = c

    def kernel(self,X,Y=None):
        """Compute the kernel k(X,Y) 
        if Y is not None, k(X,X) otherwise

        Parameters:
            X : np.array
            Y : np.array
        
        Return:
            a polynomial kernel in an array
        """
        # matrixes of inner products
        if Y is not None:
            x_tens_y = np.power((X.dot(Y.T) + self.c), self.d)
            self.k = x_tens_y
            return x_tens_y
        else:
            x_tens_y = np.power(X@X.T + self.c, self.d)
            self.k = x_tens_y
            return x_tens_y
