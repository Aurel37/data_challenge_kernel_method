import argparse
import numpy as np
import pandas as pd
import time
from kernel_pca import KernelPCA
from dataloader import DataLoader
from kernel import RBF, Polynomial
from multiclass_svm import MultiKernelSVC, Cross_validation
from utils import to_csv, transform_to_image
from compute_features import Histogram_oriented_gradient
from kmeans import KMeans

# Download the dataset

Xtr = np.array(pd.read_csv('Xtr.csv',header=None,sep=',',usecols=range(3072)))
Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('Ytr.csv',sep=',',usecols=[1])).squeeze()

# Transform the vectors into images
Xtr_im = transform_to_image(Xtr)
Xte_im = transform_to_image(Xte)

# Compute the features of the images
hog_features = np.zeros((5000, 1764))
for i in range(5000):
    hog_features[i, :] = Histogram_oriented_gradient(Xtr_im[i], cell_size=(4, 4), block_size=(2, 2), multichannel= True)

hog_features_validation = np.zeros((2000, 1764))
for i in range(2000):
    hog_features_validation[i, :] = Histogram_oriented_gradient(Xte_im[i], cell_size=(4, 4), block_size=(2, 2), multichannel= True)

parameters = (5, 0.6)
kernel = Polynomial(5, 0.6).kernel

# Do the classification
time0 = time.time()
predictions = Cross_validation(hog_features, Ytr, hog_features_validation, kernel, C= 0.1, K = 7, print_accuracy = True, parameters = parameters, return_prediction = True)
time1 = time.time()
print(" Prediction computed in {:.3f} s".format(time1 - time0))

to_csv(predictions)