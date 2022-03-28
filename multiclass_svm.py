import numpy as np
from binary_svm import KernelSVC

class MultiKernelSVC:
    """Multi Class kernel SVC
    """

    def __init__(self, C, dataloader, class_num, epsilon = 1e-15):
        self.C = C 
        # kernel should be a function here                          
        self.kernel = dataloader.kernel    
        self.K = dataloader.K
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.SVMs = []   
        self.dataloader = dataloader
        self.class_num = class_num

    def train(self):
        """train the multicclass svms using one vs all
        """
        for cl in range(self.class_num):
            svc = KernelSVC(self.C, self.kernel, self.epsilon)
            target = self.dataloader.target_train.copy()
            target[self.dataloader.target_train == cl] = 1
            target[self.dataloader.target_train != cl] = -1
            svc.fit(self.dataloader.dataset_train, target, self.K)
            self.SVMs.append(svc)

    def accuracy(self, X, y):
        n, _ = X.shape
        prediction = self.predict(X)
        return np.sum(prediction == y)/n
    
    def predict(self, X):
        n, _ = X.shape
        prediction = np.zeros(n)
        # one vs all prediction
        for cl in range(self.class_num):
            cl_prediction = self.SVMs[cl].predict(X)
            prediction[cl_prediction == 1] = cl
        return prediction