import numpy as np

class DataLoader:

    def __init__(self, dataset, target, kernel=None, prop=0.8, shuffle=False):
        if not(target.shape[0] == dataset.shape[0]):
            raise ValueError("target and dataset must have same x-axis size")
        self.N = dataset.shape[0]
        self.shuffle = self.shuffle
        self.prop = prop
        self.dataset = dataset
        self.target = target
        # self.K = self.kernel(self.dataset, self.dataset)
        # avoid to recompute the kernel at each step
        self.K = None
        # kernel should be a function here
        self.kernel = kernel
        # keep tracks of kernel to avoid compute it
        # every time
        
        if shuffle:
            arange = np.arange(self.N)
            np.random.shuffle(arange)
            self.dataset = self.dataset[arange, :]
            self.target = self.target[arange]
    
    @property
    def kernel(self):
        return self._kernel
    
    @kernel.setter
    def kernel(self, new_kernel):
        self._kernel = new_kernel
        if self.K is None and new_kernel is not None:
            self.K = new_kernel(self.dataset_train)

    @property
    def dataset(self):
        return self._dataset
    
    @dataset.setter
    def dataset(self, new_dataset):
        """update train and test every  time
        the dataset is updated
        """
        self._dataset = new_dataset.copy()
        self.dataset_train = self._dataset[:int(self.N*self.prop), :]
        self.dataset_test  = self._dataset[int(self.N*self.prop):, :]

    @property
    def target(self):
        return self._target
    
    @target.setter
    def target(self, new_target):
        """update target train and target test every  time
        the target is updated
        """
        self._target = new_target.copy()
        self.target_train = self._target[:int(self.N*self.prop)]
        self.target_test  = self._target[int(self.N*self.prop):]

    def copy_data(self):
        """copy the dataloader properly
        """
        dataset = self.dataset.copy()
        target = self.target.copy()
        return DataLoader(dataset, target, self.prop, shuffle=False)