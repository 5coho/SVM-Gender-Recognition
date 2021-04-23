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
    def __init__(self,filename,clf):

        #clf = joblib.load(f"firstTest.joblib")
        isRect = False
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        img = cv2.imread(filename)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        font = cv2.FONT_HERSHEY_SIMPLEX
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        roi_colorList = []
        savedXList = []
        savedYList = []
        for (x, y, w, h) in faces:
            # find all faces and make rectangles for them
            # make .jpg for each rectangle
            img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            savedXList.append(x)
            savedYList.append(y)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]
            roi_colorList.append(roi_color)
        data = dict()
        data['data'] = []
        data['label'] = []
        shutil.rmtree('NewJpg')
        os.mkdir('NewJpg')
        for i in range(len(roi_colorList)):
            # we write a jpg for each of the rectangle images.
            isRect = True
            cv2.imwrite('NewJpg/rulersmall'+str(i)+'.jpg',roi_colorList[i])

        current_path = os.path.join(os.getcwd(), 'NewJpg')
        i = 0
        # scott is using size 130 by 70
        # isRect is set to True in the above for loop when we have found a face in the picture
        # if no face is found then there will be no rectangle, hence we do not need to do the
        # following if it's False
        if isRect:
            for files in os.listdir(current_path):
                # if there were faces detected then read them, add
                # them to data.
                im = imread("NewJpg/rulersmall" + str(i) + ".jpg")
                im = resize(im, (150, 150))
                image = cv2.imread("NewJpg/rulersmall" + str(i) + ".jpg", cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (130, 70))
                data['data'].append(image)
                data['label'].append('male')
                i += 1
        if isRect:
            X = np.array(data['data'])
            Xori = X.reshape(X.shape[0], X.shape[1] * X.shape[2])
            y_test = np.array(data['label'])
            X_test = shuffle(Xori)
            y_pred = clf.predict(X_test)

            for i in range(len(y_pred)):
                # here we put the text of male or female on the image
                cv2.putText(img, str(y_pred[i]), (savedXList[i], savedYList[i] - 6), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imwrite('NewJpg/finalpic.jpg', img)
            self.finalImage = 'NewJpg/finalpic.jpg'
        else:
            self.finalImage = filename


    def getImage(self):

        return self.finalImage

