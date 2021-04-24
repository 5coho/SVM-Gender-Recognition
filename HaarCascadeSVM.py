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
from sklearn.svm import SVC
from sklearn.utils import shuffle


class HaarCascadeSVM:
# this class is used for the SVM
# the haarcascade class performs to haarcascade to get faces
# then it also predicts whether the face is male or female.
    def __init__(self,filename,clf):
        self.filename = filename
        self.data = dict()
        self.data['label'] = []
        self.data['filename'] = []
        self.data['data'] = []
        self.isRect = False
        self.clf = clf
        self.img = None
        self.savedXList = []
        self.savedYList = []
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def face_cascade_start(self):
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # load haar cascade, read image, make a grayscale copy
        self.img = cv2.imread(self.filename)
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        # get the faces
        roi_colorList = []
        for (x, y, w, h) in faces:
            # find all faces and make rectangles for them
            # make .jpg for each rectangle
            self.img = cv2.rectangle(self.img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self.savedXList.append(x)
            self.savedYList.append(y)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = self.img[y:y + h, x:x + w]
            roi_colorList.append(roi_color)

        # create a new folder to store the new .jpg files
        # but make sure it is empty first
        shutil.rmtree('NewJpg')
        os.mkdir('NewJpg')
        for i in range(len(roi_colorList)):
            # we write a jpg for each of the rectangle images.
            self.isRect = True
            cv2.imwrite('NewJpg/rulersmall'+str(i)+'.jpg',roi_colorList[i])

        current_path = os.path.join(os.getcwd(), 'NewJpg')
        i = 0
        
        # isRect is set to True in the above for loop when we have found a face in the picture
        # if no face is found then there will be no rectangle, hence we do not need to do the
        # following if it's False
        if self.isRect:
            for files in os.listdir(current_path):
                # if there were faces detected then read them, add
                # them to data.
                im = imread("NewJpg/rulersmall" + str(i) + ".jpg")
                im = resize(im, (150, 150))
                image = cv2.imread("NewJpg/rulersmall" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (130, 70))
                self.data['data'].append(image)
                self.data['label'].append('male')
                i += 1

    def predict_gender(self):

        if self.isRect:

            X = np.array(self.data['data'])
            Xori = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
            y_test = np.array(self.data['label'])
            X_test = shuffle(Xori)
            y_pred = self.clf.predict(X_test)
            for i in range(len(y_pred)):
                # here we put the text of male or female on the image
                cv2.putText(self.img, str(y_pred[i]), (self.savedXList[i], self.savedYList[i] - 6), self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite('NewJpg/finalpic.jpg', self.img)
            self.finalImage = 'NewJpg/finalpic.jpg'
        else:
            self.finalImage = self.filename


    def getImage(self):
        # return the image which has a rectangle over the face(s)
        # and text saying the gender
        return self.finalImage

