from kernel_pca import KernelPCA
from dataloader import DataLoader
from kernel import Linear, RBF
from multiclass_svm import MultiKernelSVC

import numpy as np
import pandas as pd

Xtr = np.array(pd.read_csv('Xtr.csv',header=None,sep=',',usecols=range(3072)))
Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('Ytr.csv',sep=',',usecols=[1])).squeeze()




# load data, DataLoader automatically divide the dataset in train
# and test with a proportion of 0.8 here
dataloader = DataLoader(Xtr[:100, :], Ytr[:100], 0.8)

# perform pca
pca = KernelPCA(dataloader, RBF(10).kernel, 99)
print('Kernel PCA ')
# project and retrieve the new dataloader with selected feature
dataloader_pca = pca.project()
print("Projected")
# multi svc
multi_svc = MultiKernelSVC(1, dataloader_pca, 10)
multi_svc.train()
accuracy = multi_svc.accuracy(dataloader_pca.dataset_test, dataloader_pca.target_test)
print(f"accuracy = {accuracy}")
