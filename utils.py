import numpy as np

def transform_to_gray(dataset):
    n, d = dataset.shape
    grayscale = np.zeros((n, 1024))
    grayscale = 0.2999*dataset[:,:1024] + 0.587*dataset[:,1024:2048] + 0.114*dataset[:,2048:3072]
    return grayscale