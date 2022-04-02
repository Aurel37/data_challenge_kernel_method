import numpy as np
import pandas as pd
import datetime


def transform_to_gray(dataset):
    n, d = dataset.shape
    grayscale = np.zeros((n, 1024))
    grayscale = 0.2999*dataset[:,:1024] + 0.587*dataset[:,1024:2048] + 0.114*dataset[:,2048:3072]
    return grayscale

def to_csv(pred):
    n = len(pred)
    date = datetime.datetime.now()
    index = np.arange(1, n+1)
    pred_format = np.array([index, pred]).T
    pd.DataFrame(pred_format, columns=["Id","Prediction"]).to_csv(f'predictions/pred_Sebastien_Aurelien_{date.strftime("%Y-%m-%d %H:%M")}.csv', index=False)