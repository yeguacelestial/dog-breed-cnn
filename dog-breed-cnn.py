# STEP 0 libraries
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
import random

# STEP 1 libraries
import cv2
import matplotlib.pyplot as plt

# STEP 0 - Import datasets
# Define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets

# STEP 1 - Detect humans
