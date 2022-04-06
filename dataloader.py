import numpy as np

class DataLoader:
    """Data structure to store all the images

    Methods:
        copy_data : deep copy of the dataset

    Attributes:
        dataset : np.array/ the original dataset
        target : np.array/ the target associated to the dataset
        test_set : np.array/ the set with unknown target
        kernel : func, the kernel associated to the data
        K : kernel(dataset_train, dataset_train), save the 
        kernel matrix to gain computation time
        dataset_train : np.array
        target_train : np.array
        dataset_validate : np.array
        target_validate : np.array
    """

    def __init__(self, dataset, target, test_set=None, kernel=None, K=None, prop=0.8):
        if not(target.shape[0] == dataset.shape[0]):
            raise ValueError("target and dataset must have same x-axis size")
        self.N = dataset.shape[0]
        self.prop = prop
        self.dataset = dataset
        self.target = target
        # self.K = self.kernel(self.dataset, self.dataset)
        # avoid to recompute the kernel at each step
        self.K = K
        # kernel should be a function here
        self.kernel = kernel
        # keep tracks of kernel to avoid compute it
        # every time

        # set with unknown label
        self.test_set = test_set
    
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
        the dataset is updated, slip the data
        into train and validate
        """
        self._dataset = new_dataset.copy()
        self.dataset_train = self._dataset[:int(self.N*self.prop), :]
        self.dataset_validate  = self._dataset[int(self.N*self.prop):, :]

    @property
    def target(self):
        return self._target
    
    @target.setter
    def target(self, new_target):
        """update target train and target test every  time
        the target is updated, slip the data
        into train and validate
        """
        self._target = new_target.copy()
        self.target_train = self._target[:int(self.N*self.prop)]
        self.target_validate  = self._target[int(self.N*self.prop):]

    def copy_data(self):
        """copy the dataloader properly
        """
        dataset = self.dataset.copy()
        target = self.target.copy()
        if self.test_set is not None:
            test_set = self.test_set.copy()
        else:
            test_set = None
        return DataLoader(dataset, target, test_set, kernel=self.kernel, K=self.K, prop=self.prop)
