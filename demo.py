from platform import architecture
from kernel_pca import KernelPCA
from dataloader import DataLoader
from kernel import Linear, RBF
from multiclass_svm import MultiKernelSVC
from utils import transform_to_gray

import numpy as np
import pandas as pd

Xtr = np.array(pd.read_csv('Xtr.csv',header=None,sep=',',usecols=range(3072)))
Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('Ytr.csv',sep=',',usecols=[1])).squeeze()




# load data, DataLoader automatically divide the dataset in train
# and test with a proportion of 0.8 here

Xtr_gray = transform_to_gray(Xtr)
dataloader = DataLoader(Xtr_gray, Ytr, 0.8, True)
# perform pca
pca = KernelPCA(dataloader, Linear().kernel, 50)
# project and retrieve the new dataloader with selected feature
dataloader_pca = pca.project()
print("Projected")
# multi svc
multi_svc = MultiKernelSVC(1, dataloader_pca, 10, one_to_one=True)
multi_svc.fit()
accuracy = multi_svc.accuracy(dataloader_pca.dataset_test, dataloader_pca.target_test)
print(f"accuracy = {accuracy}")
