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
from RGB2GrayTransformer import RGB2GrayTransformer

# utility for the training and haarcascade
# this class is modified code from https://kapernikov.com/tutorial-image-classification-with-scikit-learn/

class HogTransformer(BaseEstimator, TransformerMixin):
    #calculate histogram of oriented gradient for images

    def __init__(self, orientations,pixels_per_cell,cells_per_block, block_norm):
        self.y = None
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm

    def fit(self, X, y=None):
    # self return
        return self

    def transform(self, X, y=None):
    # here is where the gradient is calculated and
    # return an array of the HOG image
        def local_hog(X):

            hogToReturn = hog(X,orientations=self.orientations,pixels_per_cell=self.pixels_per_cell,cells_per_block=self.cells_per_block,block_norm=self.block_norm)
            return hogToReturn

        try:
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])