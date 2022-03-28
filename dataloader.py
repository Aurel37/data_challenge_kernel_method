import numpy as np

class DataLoader:

    def __init__(self, dataset, target, shuffle=False):
        if not(target.shape[0] == dataset.shape[0]):
            raise ValueError("target and dataset must have same x-axis size")
        self.N = dataset.shape[0]
        self.dataset = dataset.copy()
        self.target = target.copy()
        if shuffle:
            arange = np.arange(self.N)
            self.dataset = dataset[arange, :]
            self.target = target[arange]
        self.it = None
        self.shuffle = shuffle
      
    def __iter__(self):
        self.it = -1
        return self

    def __next__(self):
        if self.it + 1 < self.N:
            self.it += 1
            return self.dataset[self.it, :], self.target[self.it]
        else:
            raise StopIteration
    
    def copy(self):
        dataset = self.dataset.copy()
        target = self.target.copy()
        return DataLoader(dataset, target, self.shuffle)