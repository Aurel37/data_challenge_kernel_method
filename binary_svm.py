from scipy import optimize
import numpy as  np

class KernelSVC:
    
    def __init__(self, C, kernel, epsilon = 1e-3):
        self.type = 'non-linear'
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.norm_f = None
    
    def fit(self, X, y):
        #### You might define here any variable needed for the rest of the code
        N = len(y)
        diag_y = np.zeros((N,N))
        for i in range(N):
            diag_y[i,i] = y[i]
        one = np.ones(N)
        K = self.kernel(X, X)
        # Lagrange dual problem
        
        def loss(alpha):
            return  -2*alpha.T@y+alpha.T@K@alpha

        # Partial derivate of Ld on alpha
        def grad_loss(alpha):
            return -2*y + 2*K@alpha

        # Constraints on alpha of the shape :
        # -  d - C*alpha  = 0
        # -  b - A*alpha >= 0

        fun_eq = lambda alpha: alpha.T@one    
        jac_eq = lambda alpha: one
        fun_ineq_0 = lambda alpha: -alpha*y + self.C*one
        jac_ineq_0 = lambda alpha:  -diag_y
        fun_ineq_1 = lambda alpha: alpha*y
        jac_ineq_1 = lambda alpha:  diag_y
        
        constraints = ({'type': 'eq',  'fun': fun_eq, 'jac': jac_eq},
                       {'type': 'ineq', 
                        'fun': fun_ineq_0 , 
                        'jac': jac_ineq_0}, 
                       {'type': 'ineq', 
                        'fun': fun_ineq_1, 
                        'jac': jac_ineq_1}, 
                      )

        optRes = optimize.minimize(fun=lambda alpha: loss(alpha),
                                   x0=np.ones(N), 
                                   method='SLSQP', 
                                   jac=lambda alpha: grad_loss(alpha), 
                                   constraints=constraints)
        self.alpha = optRes.x
        
        # support indices
        supportIndices = np.where(np.abs(self.alpha) > self.epsilon)  
        self.alpha = self.alpha[supportIndices]
        y_support = y[supportIndices]
        self.support = X[supportIndices] #'''------------------- A matrix with each row corresponding to a support vector ------------------'''
        
        K = self.kernel(self.support, self.support)
        
        # compute b
        
        # index of alpha_i to compute b using 
        # the complementary slackness conditions
        kkt_index = np.argmax((self.alpha > 0)*(self.alpha < y_support*self.C))
        self.b = y_support[kkt_index] - self.separating_function(np.array([self.support[kkt_index]]))
        
        # compute the norm of f
        self.norm_f = np.sqrt(self.alpha.T@K@self.alpha)
        
    ### Implementation of the separting function $f$ 
    def separating_function(self,x):
        # Input : matrix x of shape N data points times d dimension
        # Output: vector of size N
        return  self.kernel(x, self.support)@self.alpha
    
    
    def predict(self, X):
        """ Predict y values in {-1, 1} """
        d = self.separating_function(X)
        return 2 * (d+self.b > 0) - 1