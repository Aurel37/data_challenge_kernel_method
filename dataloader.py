import numpy as np

class DataLoader:

    def __init__(self, dataset, target, prop=0.8, shuffle=False):
        if not(target.shape[0] == dataset.shape[0]):
            raise ValueError("target and dataset must have same x-axis size")
        self.N = dataset.shape[0]
        self.dataset = dataset.copy()
        self.target = target.copy()
        if shuffle:
            arange = np.arange(self.N)
            self.dataset = self.dataset[arange, :]
            self.target = self.target[arange]

        self.dataset_train = self.dataset[:int(self.N*prop),:]
        self.target_train = self.target[:int(self.N*prop)]

        self.dataset_test  = self.dataset[int(self.N*prop):, :]
        self.target_test  = self.target[int(self.N*prop):]

        self.shuffle = shuffle
        self.prop= prop
    
    def copy(self):
        dataset = self.dataset.copy()
        target = self.target.copy()
        return DataLoader(dataset, target, self.prop, self.shuffle)