import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import pandas as pd

import time
from datetime import timedelta

import math
import os

import scipy.misc
from scipy.stats import itemfreq
from random import sample

import pickle

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Image manipulation
import PIL.Image
from IPython.display import display

from zipfile import ZipFile
from io import BytesIO

# 2 - UNZIP FILES

# 3. RESIZE AND NORMALIZE DATA

# This function help to create  a pickle file gathering all the image from a zip folder
def DataBase_creator(archivezip, nwidth, nheight, save_name):
    # We choose the archive (zip file) + the new width and height for all the image which will be reshaped
    
    # Start-time used for printing time-usage below.
    start_time = time.time()

    # nwidth x nheight = number of features because images have nwidth x nheight pixels
    s = (len(archivezip.namelist()[:]) - 1, nwidth, nheight, 3)
    allImage = np.zeros(s)

    for i in range(1,len(archivezip.namelist()[:])):
        filename = BytesIO(archivezip.read(archivezip.namelist()[i]))
        image = PIL.Image.open(filename) # Open colour image
        image = image.resize((nwidth, nheight))
        image = np.array(image)
        image = np.clip(image/255.0, 0.0, 1.0) # 255 = max value of a pixel
        allImage[i-1] = image
    
    # We save the newly created data base
    pickle.dump(allImage, open(save_name + '.p', 'wb'))

    # Ending time
    end_time = time.time()

    # Difference between start and end-times
    time_dif = end_time - start_time

    # Print time elapsed
    print(f"Time elapsed: {str(timedelta(seconds=int(round(time_dif))))}")


# Get N most represented breeds
def main_breeds(labels_raw, Nber_breeds, all_breeds='TRUE'):
    labels_freq_pd = itemfreq(labels_raw["breed"])
    labels_freq_pd = labels_freq_pd[labels_freq_pd[:,1].argsort()[::-1]]

    if all_breeds == 'FALSE':
        main_labels = labels_freq_pd[:,0][0:Nber_breeds]
    else:
        main_labels = labels_freq_pd[:,0][:]
    
    labels_raw_np = labels_raw["breed"].to_numpy()
    labels_raw_np = labels_raw_np.reshape(labels_raw_np.shape[0],1)
    labels_filtered_index = np.where(labels_raw_np == main_labels)

    return labels_filtered_index

# Create one-hot labels
def matrix_Bin(labels):
    labels_bin = np.array([])

    labels_name, labels0 = np.unique(labels, return_inverse=True)
    labels0

    for _, i in enumerate(itemfreq(labels0)[:,0].astype(int)):
        labels_bin0 = np.where(labels0 == itemfreq(labels0)[:,0][i], 1., 0.)
        labels_bin0 = labels_bin0.reshape(1, labels_bin0.shape[0])

        if (labels_bin.shape[0] == 0):
            labels_bin = labels_bin0
        else:
            labels_bin = np.concatenate((labels_bin,labels_bin0), axis=0)
    
    print(f"Nbver SubVariables {itemfreq(labels0)[:,0].shape[0]}")
    labels_bin = labels_bin.transpose()
    print(f"Shape: {labels_bin.shape}")

    return labels_name, labels_bin