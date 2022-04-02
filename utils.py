import numpy as np

def transform_to_gray(dataset):
    n, d = dataset.shape
    grayscale = np.zeros((n, 1024))
    grayscale = 0.2999*dataset[:,:1024] + 0.587*dataset[:,1024:2048] + 0.114*dataset[:,2048:3072]
    return grayscale


def transform_to_image(dataset, nb_chanel = 3):
    n, d = dataset.shape
    d_channel = int(d / nb_chanel)
    W = int(np.sqrt(d_channel))
    dataset_mc = np.zeros((n, W, W, nb_chanel))
    for ch in range(nb_chanel):
        dataset_mc[:, :, :, ch] = dataset[:, ch*d_channel: (ch + 1) * d_channel].reshape((n, W, W))
    return dataset_mc
