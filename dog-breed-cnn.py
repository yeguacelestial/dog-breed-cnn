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

# Unzip the train and test zip file
archive_train = ZipFile("Data/test.zip", "r")
archive_test = ZipFile("Data/train.zip", "r")

# This line shows the 5 first image name of the train database
print(archive_train.namelist()[0:5])

# This line shows the number of images in the train database, 
# noted that we must remove the 1st value (column header)
print(len(archive_train.namelist()[:]) - 1)