from locale import currency
import numpy as np
from binary_svm import KernelSVC
import time

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
        #self.SVMs = {} if one_to_one else []
        self.SVMs = []
        self.dataloader = dataloader
        self.class_num = class_num
        self.one_to_one = one_to_one

    def fit(self, accuracy_print=False):
        """train the multiclass svms using one vs all
        """
        if not self.one_to_one:
            # one vs all
            for cl in range(self.class_num):
                print(f"\r cl = {cl}", end="")
                svc = KernelSVC(self.C, self.kernel, self.epsilon)
                target = self.dataloader.target_train.copy()
                target[self.dataloader.target_train == cl] = 1
                target[self.dataloader.target_train != cl] = -1
                svc.fit(self.dataloader.dataset_train, target, self.K)
                self.SVMs.append(svc)
        else:
            # one vs one
            # current_cl = 1
            # class_available = np.arange(current_cl, self.class_num)
            size = size = self.class_num*(self.class_num - 1)/2
            current_index = 1
            print("Begin Fit SVM oVo")
            for class_i  in range(self.class_num):
                
                for class_j in range(class_i + 1, self.class_num):
                    #print('#' * int((current_index)/size * 50))
                    
                    posi = np.argwhere(self.dataloader.target_train == class_j)[:, 0]
                    nega = np.argwhere(self.dataloader.target_train == class_i)[:, 0]
                    target = self.dataloader.target_train.copy()
                    target[posi] = 1
                    target[nega] = -1
                    index = np.concatenate((posi, nega))

                    train_set = self.dataloader.dataset_train[index, :]

                    target_subarray = target[index]
                    # retrieve the sub matrix
                    kernel_ij = self.K[index,:][:,index]
                    time0 = time.time()
                    svc = KernelSVC(self.C, self.kernel, self.epsilon)
                    svc.fit(train_set, target_subarray, kernel_ij)
                    time1 = time.time()
                    tps = ""
                    if class_i == 0 and class_j == 1:
                        tps = "Temps de fit : {:.2f} s.".format(time1 - time0)

                    acc = ""   
                    if accuracy_print:
                        accuracy = svc.accuracy(train_set, target_subarray)
                        acc = f"SVM ({class_i}, {class_j}) accuracy training : {accuracy:.3f}."
                    self.SVMs.append(svc)

                    print('\rProgress [{0:<50s}] current class : {1}. {2}'.format('#' * int((current_index)/size * 50), class_i+1, acc), end="")
                    current_index += 1
        print()
        # accuracy = self.accuracy(self.dataloader.dataset_train, self.dataloader.target_train)
        # print(f"accuracy training : {accuracy}")

    def accuracy(self, X, y):
        n, _ = X.shape
        prediction = self.predict(X)
        return np.sum(prediction == y)/n
    
    def predict(self, X):
        ### Inspired by the function of SckitLean for OvR Decision Function
        if self.one_to_one:
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
