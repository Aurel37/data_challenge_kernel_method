import numpy as np
from binary_svm import KernelSVC

class MultiKernelSVC:
    """Multi Class kernel SVC
    """

    def __init__(self, C, dataloader, class_num, one_to_one=False, epsilon = 1e-15):
        self.C = C 
        # kernel should be a function here                          
        self.kernel = dataloader.kernel    
        self.K = dataloader.K
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        self.SVMs = {} if one_to_one else []
        self.dataloader = dataloader
        self.class_num = class_num
        self.one_to_one = one_to_one

    def fit(self):
        """train the multiclass svms using one vs all
        """
        if not self.one_to_one:
            for cl in range(self.class_num):
                svc = KernelSVC(self.C, self.kernel, self.epsilon)
                target = self.dataloader.target_train.copy()
                target[self.dataloader.target_train == cl] = 1
                target[self.dataloader.target_train != cl] = -1
                svc.fit(self.dataloader.dataset_train, target, self.K)
                self.SVMs.append(svc)
        else:
            class_available = np.arange(1, self.class_num)
            for cl in range(self.class_num):
                print(f"\r cl = {cl}", end="")
                for cl_available in class_available:
                    n, _ =  self.dataloader.dataset_train.shape
                    posi = np.argwhere(self.dataloader.target_train == cl).T
                    nega = np.argwhere(self.dataloader.target_train == cl_available).T
                    index = np.concatenate((posi[0], nega[0]))
                    train_set = self.dataloader.dataset_train[index, :]
                    n_1 = len(posi[0])
                    n_2 = len(nega[0])
                    np.resize(index, n_1 + n_2)
                    target = np.zeros(n_1 + n_2)
                    target[:n_1] = 1
                    target[n_1:] = -1
                    kernel = np.zeros((n_1 + n_2, n))
                    kernel = self.K[index,:]
                    kernel_ij = np.zeros((n_1 + n_2, n_1 + n_2))
                    kernel_ij = kernel[:,index]
                    svc = KernelSVC(self.C, self.kernel, self.epsilon)
                    svc.fit(train_set, target, kernel_ij)
                    # side is the side from which the datapoints belongs
                    # for prediction multiply by side
                    if cl not in self.SVMs.keys():
                        self.SVMs[cl] = [svc]
                    else:
                        self.SVMs[cl].append(svc)
                # "delete" the cl studied
                class_available = np.arange(cl+1, self.class_num)

    def accuracy(self, X, y):
        n, _ = X.shape
        prediction = self.predict(X)
        return np.sum(prediction == y)/n
    
    def predict(self, X):
        n, _ = X.shape
        prediction = np.zeros(n)
        # one vs all prediction
        for cl in range(self.class_num):
            prediction_cl = np.array([True for _ in range(n)])
            for svc in self.SVMs[cl]:
                cl_prediction = svc.predict(X)
                prediction_cl = prediction_cl*(cl_prediction == 1)
            prediction[prediction_cl == 1] = cl
        return prediction