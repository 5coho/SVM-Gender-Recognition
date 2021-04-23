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
from RGB2GrayTransformer import RGB2GrayTransformer
from HogTransformer import HogTransformer


data = dict()
data['label'] = []
data['filename'] = []
data['data'] = []
src = ''
current_path = 'male'
pklname = f"somerandom.pkl"
width = 150 # scott is using 130 by 70
height = 150
i =0
maxImg = 6000
# maxImg is the number of images from each
for file in os.listdir(current_path):
    i+=1
#both will run fully
    if i == maxImg:
        break
    if file[-3:] in {'jpg'}:
        # get the pciture, resize it, add it to the data.
        im = imread(os.path.join(current_path,file))
        im = resize(im, (width, height))
        data['label'].append("male")
        data['filename'].append(file)
        data['data'].append(im)

current_path = 'female'
i = 0
for file in os.listdir(current_path):
    i+=1
#both will run fully
    if i == maxImg:
        break
    if file[-3:] in {'jpg'}:
        # get the pciture, resize it, add it to the data.
        im = imread(os.path.join(current_path,file))
        im = resize(im, (width, height))
        data['label'].append("female")
        data['filename'].append(file)
        data['data'].append(im)

joblib.dump(data,pklname)
print("resizing done")
print("time up to now = "+str(time.time()-start_time))
data = joblib.load(f"somerandom.pkl")
# makes the images grayscale
grayify = RGB2GrayTransformer()
# calculates gradients of image
hogify = HogTransformer(
    pixels_per_cell=(6, 6), #previously was 14,14, 8,8 gives better results
    cells_per_block=(2, 2),
    orientations=9,
    block_norm='L2-Hys'
)
# for scaling image
scalify = StandardScaler()

X = np.array(data['data'])
y = np.array(data['label'])
# X_train = X
# y_train = y
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.1,shuffle=True,random_state =42)
X_train_gray = grayify.fit_transform(X_train)
X_train_hog = hogify.fit_transform(X_train_gray)
X_train_prepared = scalify.fit_transform(X_train_hog)
print("test start")
print("time up to now = "+str(time.time()-start_time))
X_test_gray = grayify.fit_transform(X_test)
X_test_hog = hogify.fit_transform(X_test_gray)
X_test_prepared = scalify.transform(X_test_hog)
sgd_clf = SGDClassifier(max_iter = 2000, tol = 1e-3)
sgd_clf.fit(X_train_prepared,y_train)
# save all models for use in haarcascade class
joblib.dump(sgd_clf,'faceclassifier.joblib')
joblib.dump(scalify,'jobscalifier.pkl')
joblib.dump(grayify,'jobscalifier2.pkl')
joblib.dump(hogify,'jobscalifier3.pkl')
y_pred = sgd_clf.predict(X_test_prepared)
print(np.array(y_pred == y_test)[:25])
print('')
print('Percentage Correct: ',100*np.sum(y_pred == y_test)/len(y_test))
print(X_train_prepared.shape)
print("prgrm end")
print("Time gone by is "+str(time.time()-start_time))