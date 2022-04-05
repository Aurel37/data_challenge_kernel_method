from locale import currency
import numpy as np
from binary_svm import KernelSVC
import time
from dataloader import DataLoader

class MultiKernelSVC:
    """Multi Class kernel SVC
    """

    def __init__(self, C, dataloader, class_num, one_to_one=False, epsilon = 1e-4, verbose = -1):
        self.C = C 
        # kernel should be a function here                          
        self.kernel = dataloader.kernel    
        self.K = dataloader.K
        self.alpha = None
        self.support = None
        self.epsilon = epsilon
        #self.SVMs = {} if one_to_one else []
        self.SVMs = []
        self.dataloader = dataloader
        self.class_num = class_num
        self.one_to_one = one_to_one
        self.verbose = verbose

    def fit(self):
        """train the multiclass svms using one vs all
        """
        if not self.one_to_one:
            # one vs all
            for cl in range(self.class_num):
                if self.verbose > 0:
                    print(f"\r cl = {cl}", end="")
                svc = KernelSVC(self.C, self.kernel, self.epsilon)
                target = self.dataloader.target_train.copy()
                target[self.dataloader.target_train == cl] = 1
                target[self.dataloader.target_train != cl] = -1
                svc.fit(self.dataloader.dataset_train, target, self.K)
                self.SVMs.append(svc)
        else:
            # one vs one

            size = size = self.class_num*(self.class_num - 1)/2
            current_index = 1
            if self.verbose > 2:
                print("Begin Fit SVM oVo")
            for class_i  in range(self.class_num):
                for class_j in range(class_i + 1, self.class_num):
                    keep_idx = (self.dataloader.target_train == class_j) | (self.dataloader.target_train == class_i)
                    target = self.dataloader.target_train[keep_idx]
                    binary_target = np.ones(target.shape)
                    train_set = self.dataloader.dataset_train[keep_idx, :]

                    binary_target[target == class_i] = -1
                    kernel_ij = self.K[keep_idx,:][:,keep_idx]

                    svc = KernelSVC(self.C, self.kernel, self.epsilon)
                    svc.fit(train_set, binary_target, kernel_ij)
                    acc = ""
                    if self.verbose > 1:
                        accuracy = svc.accuracy(train_set, binary_target)
                        acc = f" SVM accuracy training : {accuracy:.3f}"
                    self.SVMs.append(svc)
                    if self.verbose > 0:
                        print('\rProgress [{0:<50s}] current class : {1}/{2}. {3}'.format('#' * int((current_index)/size * 50), class_i+1, class_j+1, acc), end="")
                    current_index += 1
        if self.verbose > 0:
            print()
    
    def accuracy(self, X, y):
        n, _ = X.shape
        prediction = self.predict(X)
        return np.sum(prediction == y)/n
    
    def predict(self, X):
        ### Inspired by the function of SckitLean for OvR Decision Function
        if self.one_to_one:
            if self.verbose > 2:
                print("Begin predict from oVo to oVo")
            n, _ = X.shape
            predictions_oVo = np.zeros((n, len(self.SVMs)))
            scores_oVo = np.zeros((n, len(self.SVMs)))

            predictions = np.zeros((n, self.class_num))
            scores = np.zeros((n, self.class_num))

            size = self.class_num*(self.class_num - 1)/2
            current_index = 0
            for class_i  in range(self.class_num):
                    for class_j in range(class_i + 1, self.class_num):
                        if self.verbose > 0:
                            print('\rProgress [{0:<50s}] current class : {1}'.format('#' * int((current_index)/size * 50), class_i+1), end="")
                        svc = self.SVMs[current_index]
                        time0 = time.time()
                        predictions_oVo[:, current_index], scores_oVo[:, current_index] = svc.predict(X, return_score = True)
                        time1 = time.time()
                        #print('Temps de calcul de la prédiction {}'.format(time1 - time0))
                        scores[:, class_i] -= scores_oVo[:, current_index]
                        scores[:, class_j] += scores_oVo[:, current_index]
                        predictions[predictions_oVo[:, current_index] == -1, class_i] += 1
                        predictions[predictions_oVo[:, current_index] == 1, class_j] += 1
                        current_index += 1
            if self.verbose > 0:
                print()
            # Here, we have for each samples the number of votes per class. 
            # We will use the scores to solve equality problems 
            # So, put scores in [-1/3; 1/3] so that is does not change the number 
            # of votes but it will influence in case of equality

            scores /= (3*(np.abs(scores) + 1))

            predictions = predictions + scores
            return np.argmax(predictions, axis= 1)

        else:
            for cl in range(self.class_num):
                cl_prediction = self.SVMs[cl].predict(X)
                predictions[cl_prediction == 1] = cl
            return predictions

def Cross_validation(Xtr, Ytr, Xte, kernel, C= 0.1, K = 7, print_accuracy = False, parameters = None, return_prediction = False):
    """
    Divide the dataset in K parts for cross validation
    Can print the average accuracy
    Can also return the prediction on Xte"""

    N = Xtr.shape[0]   
    proba = 1/K
    shift = int(N/K)
    idx = np.arange(N)
    # Shuffe the index for more generalization
    np.random.shuffle(idx)

    accuracy = 0
    SVMS = []
    # Cross Validation
    for k in range(K):
        # Shift the index for changing the validation and training set
        idx_k = (idx + k*shift)%N
        # Compute the dataloader 
        dataloader = DataLoader(Xtr[idx_k], Ytr[idx_k], kernel=kernel, prop=1 - proba, shuffle=False)
        # Do the SVM
        multi_svc = MultiKernelSVC(C, dataloader, 10, one_to_one=True, verbose = 0)
        multi_svc.fit()
        SVMS.append(multi_svc)
        # Compute accuracy of the SVM
        accuracy_k = multi_svc.accuracy(dataloader.dataset_test, dataloader.target_test)
        accuracy += accuracy_k

    accuracy /= K
    if print_accuracy :
        print(f"accuracy test = {accuracy}, with parameters (d, c) = ({parameters})")

    if return_prediction : 
        # Compute the prediction by computing prediction of each SVM, and take the main vote
        predictions = np.zeros((Xte.shape[0], K))
        for k in range(K):
            predictions[:, k] = SVMS[k].predict(Xte)
        predictions_f = np.zeros(Xte.shape[0])
        for im in range(Xte.shape[0]):
            vote = np.zeros(10)
            for k in range(K):
                vote[int(predictions[im, k])] += 1
            predictions_f[im] = np.argmax(vote)
        return predictions_f