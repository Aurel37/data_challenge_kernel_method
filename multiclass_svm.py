import numpy as np
from binary_svm import KernelSVC

class MultiKernelSVC:

    def __init__(self, C, kernel, dataloader, class_num, epsilon = 1e-3):
        self.C = C                               
        self.kernel = kernel        
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.SVMs = []   
        self.dataloader = dataloader
        self.class_num = class_num

    def train(self):
        for cl in self.class_num:
            svc = KernelSVC(self.C, self.kernel, self.epsilon)
            target = self.dataloader.target_train.copy()
            target[target == cl] = 1
            target[target != cl] = -1
            svc.fit(self.dataloader.dataset_train, target)
            self.SVMs.append(svc)
    
    def predict(self, X):
        n, _ = X.shape
        prediction = np.zeros(n)
        for cl in self.class_num:
            cl_prediction = self.SVMs[cl].predict(X)
            prediction[cl_prediction == 1] = cl
        return prediction