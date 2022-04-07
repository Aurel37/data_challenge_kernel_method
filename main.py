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

Xtr = np.array(pd.read_csv('Xtr.csv',header=None,sep=',',usecols=range(3072)))
Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('Ytr.csv',sep=',',usecols=[1])).squeeze()

if __name__ == "__main__":
    # load data, DataLoader automatically divide the dataset in train
    # and test with a proportion of 0.8 here
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pca", help="perform pca on the whole data (always after hog)", action="store_true")
    parser.add_argument("-k", "--kmeans", help="try kmeans", action="store_true")
    args = parser.parse_args()

    # test kmeans algo 
    if args.kmeans:
        dataloader_kmeans = DataLoader(Xtr, Ytr, prop=0.8)
        kmeans = KMeans(dataloader_kmeans, RBF().kernel)
        kmeans.spectral_cluestering(300)

    

    Xtr_im = transform_to_image(Xtr)
    Xte_im = transform_to_image(Xte)
    hog_features = np.zeros((5000, 1764))
    for i in range(5000):
        hog_features[i, :] = Histogram_oriented_gradient(Xtr_im[i], cell_size=(4, 4), block_size=(2, 2), method = 'L1', multichannel= True)
    
    

    hog_features_validation = np.zeros((2000, 1764))
    for i in range(2000):
        hog_features_validation[i, :] = Histogram_oriented_gradient(Xte_im[i], cell_size=(4, 4), block_size=(2, 2), method = 'L1', multichannel= True)

    dataloader = DataLoader(hog_features, Ytr, hog_features_validation, kernel=Polynomial(5, 0.6).kernel, prop=0.8)

    #perform pca
    if args.pca:
        print("Start pca")
        pca = KernelPCA(DataLoader(Xtr, Ytr, kernel=Polynomial(5, 0.6).kernel, prop=0.8), RBF().kernel, 500)
        # project and retrieve the new dataloader with selected feature
        dataloader_pca = pca.project()
        dataloader = dataloader_pca
        multi_svc = MultiKernelSVC(.1, dataloader, 10, one_to_one=True)
        multi_svc.fit()
        accuracy = multi_svc.accuracy(dataloader.dataset_validate, dataloader.target_validate)
        print(f"accuracy test with pca = {accuracy}")
    
    if False:

        # Do directly the multiclass for one split of the dataset
        time0 = time.time()
        multi_svc = MultiKernelSVC(.1, dataloader, 10, one_to_one=True)
        multi_svc.fit()
        accuracy = multi_svc.accuracy(dataloader.dataset_test, dataloader.target_test)
        print(f"accuracy test = {accuracy}")
        time1 = time.time()
        print(" Prediction computed in {:.3f} s".format(time1 - time0))

        predictions = multi_svc.predict(dataloader.validate_set)
        print(predictions)
        to_csv(predictions)

    if False:
        # If we want to do directly Cross Validation
        pred = Cross_validation(hog_features, Ytr, hog_features_validation, Polynomial(5, 0.6).kernel, C= 0.1, K = 7, print_accuracy = True, parameters = (5, 0.6), return_prediction = True)