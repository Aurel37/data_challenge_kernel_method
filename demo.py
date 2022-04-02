from platform import architecture
from kernel_pca import KernelPCA
from dataloader import DataLoader
from kernel import Linear, RBF, Polynomial
from multiclass_svm import MultiKernelSVC
from utils import transform_to_gray, transform_to_image
from compute_features import Histogram_oriented_gradient

import numpy as np
import pandas as pd
import time
from sklearn import svm
from skimage.feature import hog

Xtr = np.array(pd.read_csv('Xtr.csv',header=None,sep=',',usecols=range(3072)))
Xte = np.array(pd.read_csv('Xte.csv',header=None,sep=',',usecols=range(3072)))
Ytr = np.array(pd.read_csv('Ytr.csv',sep=',',usecols=[1])).squeeze()




# load data, DataLoader automatically divide the dataset in train
# and test with a proportion of 0.8 here
#Xtr_gray = transform_to_gray(Xtr)

Xtr_images = transform_to_image(Xtr)
N, W, H, C = Xtr_images.shape

# Xtr_gray = np.resize(Xtr_gray, (5000, 32, 32))

feat_shape = 1764
hog_features = np.zeros((N, feat_shape))
hog_ski = np.zeros((N, feat_shape))

for i in range(N):
    hhg = Histogram_oriented_gradient(Xtr_images[i], block_size=(2,2), cell_size=(4, 4), multichannel=True, flatten = True)
    hog_features[i, :] = hhg

    hog_ski[i, :] = hog(Xtr_images[i], orientations=9, pixels_per_cell=(4, 4),
                    cells_per_block=(2, 2),  block_norm = 'L1', channel_axis = -1)
    #print(np.linalg.norm(hog_features[i, :] - hog_ski[i, :]))

print('Hog is Done')

hog_ski_train = hog_features[:4000, :]
hog_ski_test = hog_features[4000:, :]

Ytr_train = Ytr[:4000]
Ytr_test = Ytr[4000:]


clist = [0.1 * i for i in range(20)]
clist = [0.3]
for c in clist:
    # clf = svm.SVC()
    # clf.fit(hog_ski_train, Ytr_train)
    # predictions = clf.predict(hog_ski_test)

    # accuracy = np.sum(predictions == Ytr_test)/len(Ytr_test)
    # print(accuracy)
    
    dataloader = DataLoader(hog_ski, Ytr, kernel=Polynomial(, c).kernel, prop=0.8, shuffle=True)

    # perform pca
    #pca = KernelPCA(dataloader, RBF().kernel, 10)
    # project and retrieve the new dataloader with selected feature
    #dataloader_pca = pca.project()
    # multi svc
    time0 = time.time()
    multi_svc = MultiKernelSVC(0.5, dataloader, 10, one_to_one=True)
    multi_svc.fit()
    accuracy = multi_svc.accuracy(dataloader.dataset_test, dataloader.target_test)
    print(f" \n accuracy test = {accuracy} for c = {c})")
    time1 = time.time()

    print(" Prediction computed in {} ".format(time1 - time0))
