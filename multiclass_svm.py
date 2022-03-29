import numpy as np
from binary_svm import KernelSVC

class MultiKernelSVC:
    """Multi Class kernel SVC
    """

    def __init__(self, C, dataloader, class_num, one_to_one=False, epsilon = 1e-4):
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
            # one vs all
            for cl in range(self.class_num):
                svc = KernelSVC(self.C, self.kernel, self.epsilon)
                target = self.dataloader.target_train.copy()
                target[self.dataloader.target_train == cl] = 1
                target[self.dataloader.target_train != cl] = -1
                svc.fit(self.dataloader.dataset_train, target, self.K)
                self.SVMs.append(svc)
        else:
            # one vs one
            current_cl = 1
            class_available = np.arange(current_cl, self.class_num)
            for cl in range(self.class_num):
                print(f"\r cl = {cl}", end="")
                for cl_available in class_available:
                    posi = np.argwhere(self.dataloader.target_train == cl).T
                    nega = np.argwhere(self.dataloader.target_train == cl_available).T
                    index = np.concatenate((posi[0], nega[0]))
                    train_set = self.dataloader.dataset_train[index, :]
                    n_1 = len(posi[0])
                    n_2 = len(nega[0])
                    target = np.zeros(n_1 + n_2)
                    target[:n_1] = 1
                    target[n_1:] = -1
                    arange = np.arange(n_1+n_2)
                    np.random.shuffle(arange)
                    target = target[arange]
                    train_set = train_set[arange, :]
                    # retrieve the sub matrix
                    kernel_ij = self.K[index,:][:,index]
                    svc = KernelSVC(self.C, self.kernel, self.epsilon)
                    svc.fit(train_set, target, kernel_ij)
                    
                    if cl not in self.SVMs.keys():
                        self.SVMs[cl] = [svc]
                    else:
                        self.SVMs[cl].append(svc)
                # "delete" the cl that will be studied
                current_cl  += 1
                class_available = np.arange(current_cl, self.class_num)

    def accuracy(self, X, y):
        n, _ = X.shape
        prediction = self.predict(X)
        return np.sum(prediction == y)/n
    
    def predict(self, X):
        n, _ = X.shape
        prediction = np.zeros(n)
        # one vs one prediction
        # class one is not in there
        for cl in range(self.class_num-1):
            prediction_boolean = np.array([True for _ in range(n)])
            # for a vector to be on a certain
            # class, it must be detected at one
            # for all svc's of the class i
            # this doesn't work only ones predicted 
            # this is ABSURD 
            for svc in self.SVMs[cl]:
                #print(svc.alpha)
                cl_prediction = svc.predict(X)
                prediction_boolean = prediction_boolean*(cl_prediction == 1)
            prediction[prediction_boolean == 1] = cl
            #print(prediction_boolean)
            #print(prediction)
        return prediction