"""

The Gui file for crack detection
loads the GUI.ui file, which is a qt5 designer file, and adds functionality.
Future:
    Error handling needs to be added. LOTS of error handling
    Image resize
    Capture Resize
    Fix how detectCrack is called --> pass parameters to Thread is ugly

__author__          = "Scott Howes"
__email__           = "showes@unbc.ca"
__version__         = "0.0.9"
__python_version__  = "3.6.8"

"""


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
        loadUi("gui/GUI.ui", self)
        self.image = None
        self.imageRecog = None
        self.cap = None
        self.figureCount = 1
        self.clf = None
        #self.cd = CrackDetection()
        self.capThread = Thread()
        self._load_connects()


    #load the connects for buttons to functions
    def _load_connects(self):
        self.bttn_load_image.clicked.connect(self.bttn_load_image_clicked)
        #self.bttn_load_clf.clicked.connect(self.bttn_load_clf_clicked)
        #self.bttn_detect.clicked.connect(self.bttn_detect_clicked)
        self.bttn_show_image.clicked.connect(self.bttn_show_image_clicked)
        #self.bttn_show_image_proc.clicked.connect(self.bttn_show_image_proc_clicked)
        #self.bttn_compare.clicked.connect(self.bttn_compare_clicked)
        self.bttn_image_save.clicked.connect(self.bttn_image_save_clicked)
        #self.bttn_image_save_proc.clicked.connect(self.bttn_image_save_proc_clicked)
        #self.bttn_detect_capture.clicked.connect(self.bttn_detect_capture_clicked)
        #self.bttn_release_cap.clicked.connect(self.bttn_release_cap_clicked)
        #self.capThread.changePixmap.connect(self.setImage)


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


    #The function that detects a cracks from an image and puts in label
    @pyqtSlot()
    def bttn_detect_clicked(self):
        threshold = self.lcd_prob.value() * 0.01

        imageCopy = self.image.copy()

        start = time.time()
        time.sleep(2)
        #self.imageCrack = self.cd.detectCrack(imageCopy, self.clf, threshold, roiSize, roiShift, process, self.check_red.isChecked(), self.check_red_rec.isChecked(), self.check_crack_prob.isChecked(), self.check_green.isChecked(), self.check_green_rec.isChecked(), self.check_smooth_prob.isChecked())
        end = time.time()

        elapsed = end - start
        self.time_label.setText("{0:.5f} s".format(elapsed))

        height, width, channel = self.imageCrack.shape
        bytesPerLine = channel * width
        qimageCrack = QImage(self.imageCrack.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

        pixmap = QPixmap(qimageCrack)
        pixmap = pixmap.scaled(self.image_proc_label.width(), self.image_proc_label.height(), Qt.KeepAspectRatio)
        self.image_proc_label.setPixmap(pixmap)
        self.image_proc_label.setAlignment(Qt.AlignCenter)


    #the function for setting up the Thread Object from GUI
    def _setThread(self):
        self.capThread.setCap(cv2.VideoCapture(int(self.combo_capture_device.currentText())))
        self.capThread.setLabel(self.capture_label)
        self.capThread.setFpsLabel(self.fps_label)
        self.capThread.setCd(self.cd)
        self.capThread.setClf(self.clf)
        self.capThread.setThreshold(self.lcd_prob.value() * 0.01)
        self.capThread.setRoiSize(int(self.combo_size.currentText()))
        self.capThread.setRoiShift(int(self.combo_shift.currentText()))
        self.capThread.setProcess(self._get_process())
        self.capThread.setRed(self.check_red.isChecked())
        self.capThread.setRedRec(self.check_red_rec.isChecked())
        self.capThread.setCrackProb(self.check_crack_prob.isChecked())
        self.capThread.setGreen(self.check_green.isChecked())
        self.capThread.setGreenRec(self.check_green_rec.isChecked())
        self.capThread.setSmoothProb(self.check_smooth_prob.isChecked())


    #The function that detects a cracks from a capture device ie) webcam and puts in label
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
        b,g,r = cv2.split(self.imageCrack)
        img2 = cv2.merge([r,g,b])
        plt.figure(self.figureCount)
        self.figureCount = self.figureCount + 1
        plt.imshow(img2)
        plt.show()


    @pyqtSlot()
    def bttn_compare_clicked(self):
        b1,g1,r1 = cv2.split(self.image)
        img1 = cv2.merge([r1,g1,b1])

        b2,g2,r2 = cv2.split(self.imageCrack)
        img2 = cv2.merge([r2,g2,b2])

        plt.figure(self.figureCount)
        self.figureCount = self.figureCount + 1
        plt.subplot(121)
        plt.imshow(img1) # expects distorted color
        plt.subplot(122)
        plt.imshow(img2) # expect true color
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
            cv2.imwrite(filePath, self.imageCrack)


#the thread class for Video Capture
#not engineered very well
class Thread(QThread):

    changePixmap = pyqtSignal(QImage)

    def __init__(self):
        super(Thread, self).__init__()
        self.cap = None
        self.label = None
        self.fpsLabel = None
        self.cd = None
        self.clf = None
        self.threshold = None
        self.roiSize = None
        self.roiShift = None
        self.process = None
        self.red = True
        self.redRec = False
        self.crackProb = False
        self.green = False
        self.greenRec = False
        self.smoothProb = False

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if ret:

                start = time.time()
                frameCrack = self.cd.detectCrack(frame, self.clf, self.threshold, self.roiSize, self.roiShift, self.process, self.red, self.redRec, self.crackProb, self.green, self.greenRec, self.smoothProb)
                end = time.time()

                if (end - start) > 0:
                    fps = 1/(end - start)
                    self.fpsLabel.setText(str(fps))
                else:
                    self.fpsLabel.setText("---------")

                height, width, channel = frameCrack.shape
                bytesPerLine = channel * width
                pixmap = QImage(frameCrack.data, width, height, bytesPerLine, QImage.Format_RGB888).rgbSwapped()

                pixmap = pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
                self.label.setAlignment(Qt.AlignCenter)
                self.changePixmap.emit(pixmap)


    #lots of setters
    def endCap(self):
        self.cap.release()
        self.label.clear()

    def setCap(self, cap):
        self.cap = cap

    def setLabel(self, label):
        self.label = label

    def setFpsLabel(self, fpsLabel):
        self.fpsLabel = fpsLabel

    def setCd(self, cd):
        self.cd = cd

    def setClf(self, clf):
        self.clf = clf

    def setThreshold(self, threshold):
        self.threshold = threshold

    def setRoiSize(self, roiSize):
        self.roiSize = roiSize

    def setRoiShift(self, roiShift):
        self.roiShift = roiShift

    def setProcess(self, process):
        self.process = process

    def setRed(self, red):
        self.red = red

    def setRedRec(self, redRec):
        self.redRec = redRec

    def setCrackProb(self, crackProb):
        self.crackProb = crackProb

    def setGreen(self, green):
        self.green = green

    def setGreenRec(self, greenRec):
        self.greenRec = greenRec

    def setSmoothProb(self, smoothProb):
        self.smoothProb = smoothProb
