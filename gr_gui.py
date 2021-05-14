"""

The Gui file Gender Recognition
loads the gender_recognition_gui.ui file and adds functionality.

"""


__author__          = "Scott Howes"
__credits__         = "Scott Howes"
__email__           = "showes@unbc.ca"
__python_version__  = "3.9.4"


#imports
import cv2
import sys
import time
from joblib import load
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtCore import QThread
from PyQt5.QtGui import QPixmap
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QWidget
from PyQt5.QtWidgets import QLabel
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtWidgets import QMessageBox
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
import numpy as np


#The GUI class
class gr_gui(QWidget):


    #the constructor
    def __init__(self):
        super(gr_gui, self).__init__()
        loadUi("gui/gender_recognition_gui.ui", self)
        self.image = None
        self.imageRecog = None
        self.cap = None
        self.figureCount = 1
        self.clf = None
        self.cascade = cv2.CascadeClassifier('haar_cascades/haarcascade_frontalface_alt2.xml')
        #self.cd = CrackDetection()
        self.capThread = Thread()
        self._load_connects()


    #load the connects for buttons to functions
    def _load_connects(self):
        self.bttn_load_image.clicked.connect(self.bttn_load_image_clicked)
        self.bttn_load_clf.clicked.connect(self.bttn_load_clf_clicked)
        self.bttn_detect.clicked.connect(self.bttn_detect_clicked)
        self.bttn_show_image.clicked.connect(self.bttn_show_image_clicked)
        self.bttn_show_image_proc.clicked.connect(self.bttn_show_image_proc_clicked)
        self.bttn_compare.clicked.connect(self.bttn_compare_clicked)
        self.bttn_image_save.clicked.connect(self.bttn_image_save_clicked)
        self.bttn_image_save_proc.clicked.connect(self.bttn_image_save_proc_clicked)
        self.bttn_detect_capture.clicked.connect(self.bttn_detect_capture_clicked)
        self.bttn_release_cap.clicked.connect(self.bttn_release_cap_clicked)
        self.capThread.changePixmap.connect(self.setImage)


    #Loads the selected image and puts into image label
    @pyqtSlot()
    def bttn_load_image_clicked(self):
        filePath, _ = QFileDialog.getOpenFileNames(None, "Load Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")

        if filePath:
            self.image = cv2.imread(filePath[0])
            self.image_filepath.setText(filePath[0])
            pixmap = QPixmap(filePath[0])
            pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), Qt.KeepAspectRatio)
            self.image_label.setPixmap(pixmap)
            self.image_label.setAlignment(Qt.AlignCenter)


    #loads the classifier
    @pyqtSlot()
    def bttn_load_clf_clicked(self):
        filePath, _ = QFileDialog.getOpenFileNames(self, "Load Classifier", "", "Joblib Files (*.joblib)")

        if filePath:
            self.clf_filepath.setText(filePath[0])
            self.clf = load(filePath[0])


    #The function that detects faces and classifies them
    #classified image is then added to image_proc_label
    @pyqtSlot()
    def bttn_detect_clicked(self):

        self.imageRecog = self.image.copy()
        imageCopyGray = cv2.cvtColor(self.imageRecog, cv2.COLOR_BGR2GRAY)

        start = time.time()

        faces = self.cascade.detectMultiScale(imageCopyGray, 1.3, 5)

        for (x,y,w,h) in faces:
            self.imageRecog = cv2.rectangle(self.imageRecog,(x,y),(x+w,y+h),(0,255,0),2)

        end = time.time()

        elapsed = end - start
        self.time_label.setText("{0:.5f} s".format(elapsed))


        height, width, channel = self.imageRecog.shape
        bytesPerLine = channel * width
        qImageCopy = QImage(self.imageRecog.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

        pixmap = QPixmap(qImageCopy)
        pixmap = pixmap.scaled(self.image_proc_label.width(), self.image_proc_label.height(), Qt.KeepAspectRatio)
        self.image_proc_label.setPixmap(pixmap)
        self.image_proc_label.setAlignment(Qt.AlignCenter)


    #the function for setting up the Thread Object from GUI
    def _setThread(self):
        self.capThread.setCap(cv2.VideoCapture(int(self.combo_capture_device.currentText())))
        self.capThread.setLabel(self.capture_label)
        self.capThread.setFpsLabel(self.fps_label)
        self.capThread.setCascade(self.cascade)
        self.capThread.setClf(self.clf)


    @pyqtSlot()
    def bttn_detect_capture_clicked(self):
        self._setThread()
        self.capThread.start()


    @pyqtSlot(QImage)
    def setImage(self, image):
        self.capture_label.setPixmap(QPixmap.fromImage(image))


    @pyqtSlot()
    def bttn_release_cap_clicked(self):
        self.capThread.endCap()
        self.capThread.terminate()


    @pyqtSlot()
    def bttn_show_image_clicked(self):
        b,g,r = cv2.split(self.image)
        img2 = cv2.merge([r,g,b])
        plt.figure(self.figureCount)
        self.figureCount = self.figureCount + 1
        plt.imshow(img2)
        plt.show()


    @pyqtSlot()
    def bttn_show_image_proc_clicked(self):
        b,g,r = cv2.split(self.imageRecog)
        img2 = cv2.merge([r,g,b])
        plt.figure(self.figureCount)
        self.figureCount = self.figureCount + 1
        plt.imshow(img2)
        plt.show()


    @pyqtSlot()
    def bttn_compare_clicked(self):
        b1,g1,r1 = cv2.split(self.image)
        img1 = cv2.merge([r1,g1,b1])

        b2,g2,r2 = cv2.split(self.imageRecog)
        img2 = cv2.merge([r2,g2,b2])

        plt.figure(self.figureCount)
        self.figureCount = self.figureCount + 1
        plt.subplot(121)
        plt.imshow(img1)
        plt.subplot(122)
        plt.imshow(img2)
        plt.show()


    @pyqtSlot()
    def bttn_image_save_clicked(self):
        filePath, _ = QFileDialog.getSaveFileName(None, "Save Image", "", "*.jpg;;*.jpeg;;*.png;;*.bmp")
        if filePath:
            cv2.imwrite(filePath, self.image)


    @pyqtSlot()
    def bttn_image_save_proc_clicked(self):
        filePath, _ = QFileDialog.getSaveFileName(None, "Save Image", "", "*.jpg;;*.jpeg;;*.png;;*.bmp")
        if filePath:
            cv2.imwrite(filePath, self.imageRecog)


#the thread class for Video Capture
class Thread(QThread):

    changePixmap = pyqtSignal(QImage)

    def __init__(self):
        super(Thread, self).__init__()
        self.cap = None
        self.label = None
        self.fpsLabel = None
        self.cascade = None
        self.clf = None


    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:

                start = time.time()

                frameCopyGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = self.cascade.detectMultiScale(frameCopyGray, 1.3, 5)

                for (x,y,w,h) in faces:
                    frame = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)


                height, width, channel = frame.shape
                bytesPerLine = channel * width
                pixmap = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

                pixmap = pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
                self.label.setAlignment(Qt.AlignCenter)
                self.changePixmap.emit(pixmap)

                end = time.time()

                if (end - start) > 0:
                    fps = 1/(end - start)
                    fps = round(fps, 4)
                    self.fpsLabel.setText(str(fps))
                else:
                    self.fpsLabel.setText("---------")

            else:
                print("something when wrong")


    #setters
    def endCap(self):
        self.cap.release()
        self.label.clear()

    def setCap(self, cap):
        self.cap = cap

    def setLabel(self, label):
        self.label = label

    def setFpsLabel(self, fpsLabel):
        self.fpsLabel = fpsLabel

    def setCascade(self, cascade):
        self.cascade = cascade

    def setClf(self, clf):
        self.clf = clf
