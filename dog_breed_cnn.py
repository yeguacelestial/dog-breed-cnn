# STEP 0 libraries
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
import random

# STEP 1 libraries
import cv2
import matplotlib.pyplot as plt

# STEP 2 libraries
from keras.applications.resnet50 import ResNet50

from keras.preprocessing import image
from PIL import Image
from tqdm import tqdm

from keras.applications.resnet50 import preprocess_input, decode_predictions
# define ResNet50 model
ResNet50_mod = ResNet50(weights='imagenet')

# STEP 3
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
# Preprocessing data
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# STEP 5
from keras.callbacks import ModelCheckpoint
from extract_bottleneck_features import *

# STEP 0 - Import datasets
# Define function to load train, test, and validation datasets
def load_dataset(path):
    data = load_files(path)
    dog_files = np.array(data['filenames'])
    dog_targets = np_utils.to_categorical(np.array(data['target']), 133)
    return dog_files, dog_targets


# STEP 1 - Detect humans
# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# STEP 2 - Detect dogs
def path_to_tensor(img_path):
    # Loads RGB image as PIL.Image.Image type
    img = Image.open(img_path)
    img = img.resize((224,224))

    # Convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)

    # Convert 3D Tensor to 4D Tensor with shape (1, 224, 224, 3) and return 4D Tensor
    return np.expand_dims(x, axis=0)

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

def ResNet50_predict_labels(img_path):
    # Returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_mod.predict(img))

# Returns true if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))


# Function that takes path of an image as an input and returns the dog breed predicted by the model
def classify_dog_breed(img_path, ResNet50_model):
    img = path_to_tensor(img_path)
    bottleneck_feature = extract_Resnet50(img)

    predictions = ResNet50_model.predict(bottleneck_feature)
    prediction = np.argmax(predictions)
    dog_names[prediction].split('.')[-1]
    print(f"This image looks like a {dog_names[prediction].split('.')[-1]}")
    return dog_names[prediction].split('.')[-1]
