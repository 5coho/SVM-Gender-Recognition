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

class haarcascade():
# this class is for the SGD
# the haarcascade class performs to haarcascade to get faces
# then it also predicts whether the face is male or female.
    def __init__(self,filename):

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        # load haar cascade, read image, make a grayscale copy
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        faces = face_cascade.detectMultiScale(gray,1.3,5)
        roi_colorList = []
        savedXList = []
        savedYList = []

        for (x,y,w,h) in faces:
            # find all faces and make rectangles for them
            # make .jpg for each face in rectangle
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            savedXList.append(x)
            savedYList.append(y)
            roi_gray = gray[y:y+h,x:x+w]
            roi_color = img[y:y+h,x:x+w]
            roi_colorList.append(roi_color)

        scalify = joblib.load(f'jobscalifier.pkl')
        data = dict()
        data['label'] = []
        data['filename'] = []
        data['data'] = []
        shutil.rmtree('NewJpg')
        os.mkdir('NewJpg')
        isRect = False
        for i in range(len(roi_colorList)):
            # we write a jpg for each of the rectangle images.
            isRect = True
            cv2.imwrite('NewJpg/rulersmall'+str(i)+'.jpg',roi_colorList[i])
        # load classifier
        clf = joblib.load(f'faceclassifier.joblib')
        current_path = os.path.join(os.getcwd(),'NewJpg')
        i = 0
        # scott is using size 130 by 70
        if isRect:
            for files in os.listdir(current_path):
                # if there were faces detected then read them, add
                # them to data.
                im = imread("NewJpg/rulersmall"+str(i)+".jpg")
                im = resize(im,(150,150))
                data['label'].append("male")
                data['filename'].append("rulersmall"+str(i)+".jpg")
                data['data'].append(im)
                i+=1
        # load classifier utilities
        grayify = joblib.load(f'jobscalifier2.pkl')
        hogify = joblib.load(f'jobscalifier3.pkl')

        if isRect:
            X = np.array(data['data'])
            y = np.array(data['label'])
            X_test = X
            y_test = y
            # for the SGD we have to make it gray, do the gradient, and
            # scale it
            X_test_gray = grayify.fit_transform(X_test)
            X_test_hog = hogify.fit_transform(X_test_gray)
            X_test_prepared = scalify.transform(X_test_hog)
            y_pred = clf.predict(X_test_prepared)

            for i in range(len(y_pred)):
                # write text on each image saying male or female
                cv2.putText(img, str(y_pred[i]), (savedXList[i], savedYList[i] - 6), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            # write image to disk
            cv2.imwrite('NewJpg/finalpic.jpg',img)
            self.finalImage = 'NewJpg/finalpic.jpg'

        else:
            self.finalImage = filename


    def getImage(self):
    # return the image which has a rectangle over the face(s)
    # and text saying the gender
        return self.finalImage

