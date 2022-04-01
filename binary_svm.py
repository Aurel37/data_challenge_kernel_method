from scipy import optimize
import numpy as  np
import cvxpy as cp

class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
    
    def fit(self, X, y, K):
        #### You might define here any variable needed for the rest of the code
        N = len(y)
        diag_y = np.zeros((N,N))
        for i in range(N):
            diag_y[i,i] = y[i]
        one = np.ones(N)
        # Lagrange dual problem
        alpha = cp.Variable(N)
        prob= cp.Problem(cp.Minimize(-2*alpha.T@y + cp.quad_form(alpha, K)), [alpha.T@one == 0, cp.multiply(y, alpha) - self.C*one <= 0, -cp.multiply(y, alpha)  <= 0])
        prob.solve()
        self.alpha = alpha.value
        # support indices
        supportIndices = np.where(np.abs(self.alpha) > self.epsilon)  
        self.alpha = self.alpha[supportIndices]
        y_support = y[supportIndices]
        self.support = X[supportIndices] #'''------------------- A matrix with each row corresponding to a support vector ------------------'''
        
        # compute b
        
        # index of alpha_i to compute b using 
        # the complementary slackness conditions
        kkt_index = np.argmax((self.alpha > 0)*(self.alpha < y_support*self.C))
        self.b = y_support[kkt_index] - self.separating_function(np.array([self.support[kkt_index]]))
        
    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return  self.kernel(x, self.support)@self.alpha
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b > 0) - 1