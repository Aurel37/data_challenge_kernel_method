import numpy as  np
import cvxpy as cp
from cvxopt import matrix
from cvxopt import solvers

class KernelSVC:
    """Binary kernel SVM
    
    Attributes:
        C : float, regularization parameter
        kernel : func
        alpha : np.array, prediction vector
        support : np.array, support vectors
        epsilon : float, selection parameters
        for the support vectors
        solver : string, "cvxopt" or "cp"
    """
    
    def __init__(self, C, kernel, epsilon = 1e-3, solver="cvxopt"):
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.solver = solver
    
    def fit(self, X, y, K):
        """fit the svm (inspired of the code used during the homework)
        the main difference is about the optimizer used

        Parameters:
            X : np.array/ dataset
            y : np.array/ target
            K : np.array/ the gram matrix 
            (to speed up computation already computed)
        """
        N = len(y)
        diag_y = np.zeros((N,N))
        for i in range(N):
            diag_y[i,i] = y[i]
        one = np.ones(N)
        # Lagrange dual problem
        if self.solver == "cvxpy":
            alpha = cp.Variable(N)
            # 1/2 * alpha.T @ diag_y @ K @ diag_y @alpha - np.ones(N).T@alpha
            # -2*alpha.T@y + cp.quad_form(alpha, K)
            prob= cp.Problem(cp.Minimize(-2*alpha.T@y + cp.quad_form(alpha, K)), [alpha.T@one == 0, diag_y@alpha - self.C*one <= 0, -diag_y@alpha  <= 0])
            prob.solve()
            self.alpha = alpha.value
        else:
            # use cvxopt.qp, fast optimizer
            y_cast = y.astype('float')
            P = matrix(K)
            q = matrix(-y_cast)
            
            h = matrix(self.C, (N,1))
            A = matrix(1.0, (1,N))
            b = matrix(0.)
            
            
            G = np.zeros((2*N, N))
            G[:N,:] = diag_y
            G[N:,:] = -diag_y
            G = matrix(G)
            h = np.zeros(2*N)
            h[:N] = self.C*one
            h = matrix(h)
            solvers.options['show_progress'] = False
            alpha = solvers.qp(P,q,G,h,A,b)
            self.alpha = np.ravel(alpha['x']) 

        # support indices
        supportIndices = np.where(np.abs(self.alpha) > self.epsilon)  
        self.alpha = self.alpha[supportIndices]
        y_support = y[supportIndices]
        self.support = X[supportIndices]
        # compute b
        # index of alpha_i to compute b using 
        # the complementary slackness conditions
        kkt_index = np.argmax((self.alpha > 0)*(self.alpha < y_support*self.C))
        self.b = y_support[kkt_index] - self.separating_function(np.array([self.support[kkt_index]]))
        
    def separating_function(self,x):
        """Implementation of the separting function f(x)

        Parameters:
            x : np.array
        """
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return  self.kernel(x, self.support)@self.alpha
    
    def predict(self, X, return_score = False):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        if return_score:
            return 2 * (d+self.b > 0) - 1, d
        else:
            return 2 * (d+self.b > 0) - 1


    def accuracy(self, X, y):
        """compute accuracy"""
        n, _ = X.shape
        prediction = self.predict(X)
        return np.sum(prediction == y)/n
