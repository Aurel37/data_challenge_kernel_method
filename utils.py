import numpy as np
import pandas as pd
import datetime

def transform_to_gray(dataset):
    """ grey scaling of RGB images """
    n, d = dataset.shape
    grayscale = np.zeros((n, 1024))
    grayscale = 0.2999*dataset[:,:1024] + 0.587*dataset[:,1024:2048] + 0.114*dataset[:,2048:3072]
    return grayscale


def transform_to_image(dataset, nb_chanel = 3):
    """ Restore the 3D dimension of the images of the dataset """ 
    n, d = dataset.shape
    d_channel = int(d / nb_chanel)
    W = int(np.sqrt(d_channel))
    dataset_mc = np.zeros((n, W, W, nb_chanel))
    for ch in range(nb_chanel):
        dataset_mc[:, :, :, ch] = dataset[:, ch*d_channel: (ch + 1) * d_channel].reshape((n, W, W))
    return dataset_mc

    
def to_csv(pred):
    """ Convert the prediction to csv """
    n = len(pred)
    date = datetime.datetime.now()
    index = np.arange(1, n+1)
    pred_format = np.array([index, pred],dtype=int).T
    pd.DataFrame(pred_format, columns=["Id","Prediction"]).to_csv(f'predictions/pred_Sebastien_Aurelien_{date.strftime("%Y-%m-%d %H:%M")}.csv', index=False)
