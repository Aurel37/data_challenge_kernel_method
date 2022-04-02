from platform import architecture
from kernel_pca import KernelPCA
from dataloader import DataLoader
from kernel import Linear, RBF, Polynomial
from multiclass_svm import MultiKernelSVC
from utils import transform_to_gray, to_csv
from compute_features import Histogram_oriented_gradient

import numpy as np
import pandas as pd
import time

Xtr = np.array(pd.read_csv('Xtr.csv',header=None,sep=',',usecols=range(3072)))
Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('Ytr.csv',sep=',',usecols=[1])).squeeze()



if __name__ == "__main__":
    # load data, DataLoader automatically divide the dataset in train
    # and test with a proportion of 0.8 here
    Xtr_gray = transform_to_gray(Xtr)

    Xtr_gray = np.resize(Xtr_gray, (5000, 32, 32))
    #hog_xtr = hog(Xtr_gray, cells_per_block=(1,1), pixels_per_cell=(32, 32), feature_vector=True)

    hog_features = np.zeros((5000, 324))
    for i in range(5000):
        hhg = Histogram_oriented_gradient(Xtr_gray[i], block_size=(2,2), cell_size=(8, 8), flatten = True)
        hog_features[i, :] = hhg

    dataloader = DataLoader(hog_features, Ytr, kernel=Polynomial(6, 1).kernel, prop=0.8, shuffle=True)

    # perform pca
    #pca = KernelPCA(dataloader, RBF().kernel, 10)
    # project and retrieve the new dataloader with selected feature
    #dataloader_pca = pca.project()
    # multi svc
    time0 = time.time()
    multi_svc = MultiKernelSVC(0.5, dataloader, 10, one_to_one=True)
    multi_svc.fit()
    accuracy = multi_svc.accuracy(dataloader.dataset_test, dataloader.target_test)
    print(f"accuracy test = {accuracy}")
    time1 = time.time()
    print(" Prediction computed in {}".format(time1 - time0))


    Xte_gray = transform_to_gray(Xte)
    Xte_gray = np.resize(Xte_gray, (2000, 32, 32))
    #hog_xtr = hog(Xtr_gray, cells_per_block=(1,1), pixels_per_cell=(32, 32), feature_vector=True)

    hog_features = np.zeros((2000, 324))
    for i in range(2000):
        hhg = Histogram_oriented_gradient(Xte_gray[i], block_size=(2,2), cell_size=(8, 8), flatten = True)
        hog_features[i, :] = hhg
    predictions = multi_svc.predict(hog_features)
    print(predictions)
    to_csv(predictions)
