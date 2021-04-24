import numpy as np
import cv2
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split
import skimage
import numpy as np
from skimage.feature import hog
from skimage.io import imread
from skimage.transform import rescale
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import time
start_time = time.time()
import pprint
import pickle
import shutil

# utility for the training and haarcascade
# this class is modified code from https://kapernikov.com/tutorial-image-classification-with-scikit-learn/

class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
# the class converts colored images into grayscale
    def __init__(self):
        pass

    def fit(self, X, y=None):
    # self return
        return self

    def transform(self, X, y=None):
    # here is where the conversion happens and return result
        return np.array([skimage.color.rgb2gray(img) for img in X])