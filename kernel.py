import numpy as np

class RBF:
    def __init__(self, sigma=1.):
        self.sigma = sigma  ## the variance of the kernel
        #self.k = None
        
    def kernel(self,X,Y):
        ## Input vectors X and Y of shape Nxd and Mxd
        
        # difference between all vectors X and Y
        xmy = X[:, None, :] - Y[None, :, :]
        #self.k = xmy
        return np.exp(-1/2*np.square(np.linalg.norm(xmy, axis=2)/self.sigma))

class Linear:
    def __init__(self):
        self.k = None
    
    def kernel(self,X,Y):
        # matrixes of inner products
        x_tens_y = X[:,:]@Y[:, :].T
        self.k = x_tens_y
        return x_tens_y